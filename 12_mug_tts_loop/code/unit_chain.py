"""Multi-chunk unit measurement end to end, on Amax (Claude, 2026-11-22; PROTOCOL addendum 3). System python3.
 1. send the code + unit_states.json to the cluster and check their sha256 on both machines;
 2. launch unit_run.py on Amax (smi 4-8, states j % 9 in 0..4) and on the cluster (0-3, j % 9 in 5..8);
    13:55 hand-off: the tails of both queues run on the PRO 6000s (unit_run_bw.py, unit_reassign.json "to_bw");
    the chain then also waits for UNIT_DONE_<host>-bw.json, audits those logs, and skips cluster rows of moved states;
 3. wait for both UNIT_DONE_<host>.json; audit every job log for render errors (OIDN / DeviceLost / vk:: / cannot
    create / out of memory): such jobs are excluded (excluded.json) and reported, not rerun;
 4. pull the cluster's rows and recordings into Amax's unit_measure (rows merged by (decision, cand));
 5. PRM features (unit_prm.py) with prm_service.py on one fully idle Amax card among 4-8, then stop the service;
 6. unit_analysis.py -> unit_result.json / unit_result.txt; write UNIT_ALL_DONE.
Never touches other users' processes. Log: r0/unit_measure/unit_chain.log.
usage: setsid nohup python3 code/unit_chain.py [--resume] > /dev/null 2>&1 < /dev/null &
  --resume: skip steps 1-2.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

M = Path("/workspace/fact_local/mug_loop_1121"); RD = M / "r0"; OD = RD / "unit_measure"
CL = "$USER@CLUSTER_HOST"; CM = "/workspace_cluster/fact_local/mug_loop_1121"; CRD = CM + "/r0"; COD = CRD + "/unit_measure"
PY = "/workspace/miniconda3/envs/RoboTwin/bin/python"
PRMPY = "/workspace/venvs/prm_dopamine/bin/python"; P = "/workspace/prm"
CODE = ["continue_record.py", "unit_run.py", "unit_common.py", "prm_common.py", "exec_candidates.py"]
BAD = r"OIDN|DeviceLost|vk::|cannot create|out of memory"
LOG = OD / "unit_chain.log"


def log(msg):
    with LOG.open("a") as f:
        f.write(time.strftime("%F %T ") + msg + "\n")


def sh(cmd, timeout=600):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except subprocess.TimeoutExpired:
        log(f"command timed out after {timeout}s (ignored): {cmd[:120]}"); return ""


def ssh(cmd, timeout=600):
    return sh(f"ssh -n -o BatchMode=yes {CL} '{cmd}'", timeout)


def idle(g):
    apps = sh(f"nvidia-smi -i {g} --query-compute-apps=pid --format=csv,noheader")
    used = sh(f"nvidia-smi -i {g} --query-gpu=memory.used --format=csv,noheader,nounits")
    return apps == "" and used.isdigit() and int(used) < 1024


def launch():
    ssh(f"mkdir -p {COD}")
    sh(f"scp -q {' '.join(str(M / 'code' / f) for f in CODE)} {CL}:{CM}/code/")
    sh(f"scp -q {OD / 'unit_states.json'} {CL}:{COD}/")
    files = [str(M / "code" / f) for f in CODE] + [str(OD / "unit_states.json")]
    a = sh("sha256sum " + " ".join(files) + " | cut -c1-64")
    b = ssh("sha256sum " + " ".join(f.replace(str(M), CM) for f in files) + " | cut -c1-64")
    if not a or a != b:
        log(f"code/state sha mismatch between machines; abort\n{a}\n{b}"); raise SystemExit(1)
    log(f"code + states identical on both machines ({len(files)} files)")
    # "cd X; setsid ... &" (with && the ssh never returns, 11-22 10:23)
    sh(f"cd {M}; setsid nohup python3 code/unit_run.py --machine amax --round-dir {RD} --out-dir {OD} --gpus 4,5,6,7,8 "
       f"--mod 9 --rems 0,1,2,3,4 > /dev/null 2>&1 < /dev/null &", 60)
    ssh(f"cd {CM}; setsid nohup python3 code/unit_run.py --machine cluster --round-dir {CRD} --out-dir {COD} "
        f"--gpus 0,1,2,3 --mod 9 --rems 5,6,7,8 > /dev/null 2>&1 < /dev/null &", 60)
    log("unit_run launched on Amax (4-8) and cluster (0-3)")
    time.sleep(300)
    a_ready = sh(f"grep -c 'server ready' {OD}/unit_run_Amax-Server.log 2>/dev/null || true")
    c_ready = ssh(f"grep -c \"server ready\" {COD}/unit_run_user.log 2>/dev/null || true")
    log(f"servers ready after 5 min: Amax {a_ready}, cluster {c_ready}")


def main():
    OD.mkdir(parents=True, exist_ok=True)
    resume = "--resume" in sys.argv
    log("start" + (" (resume)" if resume else ""))
    if not resume:
        launch()
    t0 = time.time()
    while True:
        a_done = (OD / "UNIT_DONE_Amax-Server.json").exists()
        c_done = ssh(f"ls {COD}/UNIT_DONE_user.json 2>/dev/null") != ""
        bw_done = (OD / "UNIT_DONE_Amax-Server-bw.json").exists() or not (OD / "unit_reassign.json").exists()
        if a_done and c_done and bw_done:
            break
        if time.time() - t0 > 8 * 3600:
            log("timeout waiting for unit_run (8 h); abort"); return
        time.sleep(120)
    log("all runners done: Amax " + (OD / "UNIT_DONE_Amax-Server.json").read_text().strip()
        + " | cluster " + ssh(f"cat {COD}/UNIT_DONE_user.json")
        + (" | PRO 6000 " + (OD / "UNIT_DONE_Amax-Server-bw.json").read_text().strip()
           if (OD / "UNIT_DONE_Amax-Server-bw.json").exists() else ""))
    # 13:55 hand-off (user: "能否在Pro6000上也并行"): states moved to the PRO 6000 runner (unit_run_bw.py)
    bw = set()
    if (OD / "unit_reassign.json").exists():
        bw = {tuple(x) for x in json.loads((OD / "unit_reassign.json").read_text())["to_bw"]}
        log(f"states run by the PRO 6000 runner (fixed by queue position): {sorted(bw)}")
    # audit
    st = [(x["seed"], x["decision"]) for x in json.loads((OD / "unit_states.json").read_text())["states"]]
    tainted = []
    for j, (s, d) in enumerate(st):
        for K in (1, 2, 3):
            if (s, d) in bw:
                n = sh(f"grep -ciE '{BAD}' {OD}/logs_Amax-Server-bw/seed{s}_d{d}_u{K}.log 2>/dev/null || true")
            elif j % 9 in {0, 1, 2, 3, 4}:
                n = sh(f"grep -ciE '{BAD}' {OD}/logs_Amax-Server/seed{s}_d{d}_u{K}.log 2>/dev/null || true")
            else:
                n = ssh(f"grep -ciE \"{BAD}\" {COD}/logs_user/seed{s}_d{d}_u{K}.log 2>/dev/null || true")
            if n and n.split()[0].isdigit() and int(n.split()[0]) > 0:
                tainted.append((s, d, K))
    (OD / "excluded.json").write_text(json.dumps(tainted))
    log(f"audit: render-error jobs (excluded) {tainted}")
    # pull + merge
    pull = OD / "_cluster"; (pull / "rec").mkdir(parents=True, exist_ok=True)
    sh(f"scp -q '{CL}:{COD}/seed*.unit*.jsonl' {pull}/", 1800)
    sh(f"scp -q '{CL}:{COD}/rec/*.npz' {pull}/rec/", 3600)
    n_rows = 0
    for f in pull.glob("seed*.unit*.jsonl"):
        dst = OD / f.name
        have = set()
        if dst.exists():
            have = {(r["decision"], r["cand"]) for r in map(json.loads, dst.read_text().splitlines()) if r}
        seed_ = int(f.name[4:10])
        new = [l for l in f.read_text().splitlines() if l.strip() and (json.loads(l)["decision"], json.loads(l)["cand"]) not in have
               and (seed_, json.loads(l)["decision"]) not in bw]
        with dst.open("a") as fo:
            for l in new:
                fo.write(l + "\n"); n_rows += 1
    n_npz = 0
    for f in (pull / "rec").glob("*.npz"):
        s_, d_ = int(f.name[4:10]), int(f.name.split("_d")[1].split("_")[0])
        if (s_, d_) in bw:
            continue
        if not (OD / "rec" / f.name).exists():
            f.rename(OD / "rec" / f.name); n_npz += 1
    log(f"cluster merged: {n_rows} rows, {n_npz} recordings")
    # PRM
    g = next((g for g in (4, 5, 6, 7, 8) if idle(g)), None)
    if g is None:
        log("no idle Amax card for the PRM; skipping PRM features")
    else:
        svc = (f"cd {M}/code; HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 "
               f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={g} PYTHONUNBUFFERED=1 "
               f"setsid {PRMPY} prm_service.py --model {P}/models/GRM-2.0-8B-Preview --port 8650 "
               f"--prompt-src {P}/Robo-Dopamine/examples/inference.py --blank-goal {P}/Robo-Dopamine/examples/blank_goal.png "
               f"> {OD}/prm_service.log 2>&1 < /dev/null & echo $!")
        spid = sh(svc, 60).split()[-1]
        log(f"PRM service pid {spid} on gpu {g}")
        for _ in range(90):
            if "PRM_SERVICE_READY" in (OD / "prm_service.log").read_text(errors="ignore"):
                break
            time.sleep(20)
        if "PRM_SERVICE_READY" in (OD / "prm_service.log").read_text(errors="ignore"):
            out = sh(f"cd {M}/code; {PRMPY} unit_prm.py --round-dir {RD} --out-dir {OD} > {OD}/unit_prm.out 2>&1; tail -1 {OD}/unit_prm.out", 4 * 3600)
            log(f"PRM features: {out}")
        else:
            log("PRM service not ready after 30 min; skipping PRM features")
        pg = sh(f"ps -o pgid= -p {spid} | tr -d ' '")
        if pg:
            sh(f"kill -- -{pg}; sleep 10; kill -9 -- -{pg} 2>/dev/null; true", 60)
        log("PRM service stopped")
    # analysis
    out = sh(f"cd {M}/code; {PY} unit_analysis.py --round-dir {RD} --out-dir {OD} > {OD}/unit_analysis.out 2>&1; tail -3 {OD}/unit_analysis.out", 3600)
    log(f"analysis: {out}")
    (OD / "UNIT_ALL_DONE").write_text(time.strftime("%F %T") + "\n")
    log("done")


if __name__ == "__main__":
    main()
