"""Independent gate re-test, end to end, on Amax (Claude, 2026-11-22; PROTOCOL addendum 2). System python3.
 1. archive the first gate's done markers / run logs; send gate_pairs_retest.json to the cluster;
 2. launch gate_run.py on Amax (smi 4-8) and on the cluster (0-3) via ssh, one server + one driver per card,
    states split j % 9 in {0..4} (Amax) / {5..8} (cluster);
 3. wait for both GATE_CONT_DONE files; audit every re-test state's driver log for render errors
    (OIDN / DeviceLost / vk:: / cannot create / out of memory): such a state's rows are archived and the state is
    rerun once on Amax (fresh log); still tainted or incomplete -> its rows are archived and it is reported/excluded;
 4. merge the cluster's re-test rows into Amax's <ep>.cont.jsonl (re-test states only, tainted states excluded);
 5. gate_eval.py (frozen critic) on the 226 states -> r0/gate_retest_result.json (decision), and on the pooled 296
    -> r0/gate_pooled296_result.json (descriptive); touch r0/GATE_RETEST_DONE.
Never touches other users' processes. Log: r0/gate_retest_chain.log.
usage: setsid nohup python3 gate_retest_chain.py [--resume] > /dev/null 2>&1 < /dev/null &
  --resume: skip steps 1-2 (gate runs already launched), go straight to waiting.
If r0/gate_retest_reassign.json exists, its amax_extra states count as Amax states (audit Amax log, no cluster merge).
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

M = Path("/workspace/fact_local/mug_loop_1121"); RD = M / "r0"
CL = "$USER@CLUSTER_HOST"; CM = "/workspace_cluster/fact_local/mug_loop_1121"; CRD = CM + "/r0"
PY = "/workspace/miniconda3/envs/RoboTwin/bin/python"
PAIRS = "gate_pairs_retest.json"
BAD = r"OIDN|DeviceLost|vk::|cannot create|out of memory"
LOG = RD / "gate_retest_chain.log"
ARCH = RD / "_gate_retest_archive"


def log(msg):
    with LOG.open("a") as f:
        f.write(time.strftime("%F %T ") + msg + "\n")


def sh(cmd, timeout=600):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"command timed out after {timeout}s (ignored): {cmd[:120]}")
        return ""
    return r.stdout.strip()


def ssh(cmd, timeout=600):
    return sh(f"ssh -n -o BatchMode=yes {CL} '{cmd}'", timeout)


def states():
    return [(s["seed"], s["decision"]) for s in json.loads((RD / PAIRS).read_text())["states"]]


def cont_rows(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def drop_states(files, bad):
    """Remove rows of the given (seed, decision) states from cont files (originals copied to ARCH first)."""
    for f in files:
        f = Path(f)
        if not f.exists():
            continue
        seed = int(f.name[16:22])
        rows = cont_rows(f)
        keep = [r for r in rows if (seed, r["decision"]) not in bad]
        if len(keep) != len(rows):
            shutil.copy(f, ARCH / f"{f.name}.{time.strftime('%H%M%S')}")
            f.write_text("".join(json.dumps(r) + "\n" for r in keep))


def main():
    ARCH.mkdir(exist_ok=True)
    resume = "--resume" in sys.argv
    log("start" + (" (resume: archive/launch already done, waiting)" if resume else ""))
    if not resume:
        launch()
    wait_and_finish()


def launch():
    # 1. archive the first gate's markers/logs (both machines), send the state list
    for p in list(RD.glob("GATE_CONT_DONE_*.json")) + list(RD.glob("gate_run_Amax-Server.log")):
        p.rename(ARCH / ("gate1_" + p.name))
    ssh(f"cd {CRD} && mkdir -p _gate_retest_archive && for f in GATE_CONT_DONE_*.json gate_run_user.log; do "
        f"[ -e $f ] && mv $f _gate_retest_archive/gate1_$f; done; true")
    sh(f"scp -q {RD / PAIRS} {CL}:{CRD}/")
    a, b = sh(f"sha256sum {RD / PAIRS} | cut -c1-64"), ssh(f"sha256sum {CRD}/{PAIRS} | cut -c1-64")
    if a != b:
        log(f"state list sha mismatch {a} {b}; abort"); raise SystemExit(1)
    # 2. launch both machines
    # "cd X; setsid ... &" (not "cd X && setsid ... &": that backgrounds a subshell which keeps ssh open, 11-22 10:23)
    sh(f"cd {M}; setsid nohup python3 code/gate_run.py --machine amax --round-dir {RD} --gpus 4,5,6,7,8 --per-gpu 1 "
       f"--pairs-file {PAIRS} --mod 9 --rems 0,1,2,3,4 > /dev/null 2>&1 < /dev/null &", 60)
    ssh(f"cd {CM}; setsid nohup python3 code/gate_run.py --machine cluster --round-dir {CRD} --gpus 0,1,2,3 "
        f"--per-gpu 1 --pairs-file {PAIRS} --mod 9 --rems 5,6,7,8 > /dev/null 2>&1 < /dev/null &", 60)
    log("gate_run launched on Amax (4-8) and cluster (0-3)")
    time.sleep(300)
    a_ready = sh(f"grep -c 'server ready' {RD}/gate_run_Amax-Server.log 2>/dev/null || true")
    c_ready = ssh(f"grep -c \"server ready\" {CRD}/gate_run_user.log 2>/dev/null || true")
    log(f"servers ready after 5 min: Amax {a_ready}, cluster {c_ready}")
    if not (a_ready and int(a_ready) > 0 and c_ready and int(c_ready) > 0):
        log("a machine has no ready server (cards busy?); abort -- states of that machine would never run")
        raise SystemExit(1)


def wait_and_finish():
    # 3. wait
    t0 = time.time()
    while True:
        a_done = (RD / "GATE_CONT_DONE_Amax-Server.json").exists()
        c_done = ssh(f"ls {CRD}/GATE_CONT_DONE_user.json 2>/dev/null") != ""
        if a_done and c_done:
            break
        if time.time() - t0 > 8 * 3600:
            log("timeout waiting for gate_run (8 h); abort"); return
        time.sleep(120)
    log("both machines done")
    st = states()
    mine = {s for j, s in enumerate(st) if j % 9 in {0, 1, 2, 3, 4}}
    ra = RD / "gate_retest_reassign.json"        # 12:40 hand-off: tail of the cluster queue run on Amax
    if ra.exists():
        extra = {tuple(x) for x in json.loads(ra.read_text())["amax_extra"]}
        mine |= extra
        log(f"reassigned to Amax (assignment fixed by queue position): {sorted(extra)}")
    tainted = []
    for s, d in st:
        if (s, d) in mine:
            n = sh(f"grep -ciE '{BAD}' {RD}/gate_logs_Amax-Server/seed{s}_d{d}.log 2>/dev/null || true")
        else:
            n = ssh(f"grep -ciE \"{BAD}\" {CRD}/gate_logs_user/seed{s}_d{d}.log 2>/dev/null || true")
        if n and int(n.split()[0]) > 0:
            tainted.append((s, d))
    srv = sh("grep -ci error /workspace/fact_local/runs_server/server_gate_r0_gpu*_s0/server.log | "
             "awk -F: '{s+=$2} END {print s}'")
    srv_c = ssh(f"grep -ci error {CM}/runs_server/server_gate_r0_gpu*_s0/server.log | awk -F: \"{{s+=\\$2}} END {{print s}}\"")
    log(f"audit: render-error states {tainted}; server error lines Amax {srv} cluster {srv_c}")
    # 4. pull the cluster rows, merge re-test states only
    pull = RD / "_cluster_cont_retest"; pull.mkdir(exist_ok=True)
    sh(f"scp -q '{CL}:{CRD}/hanging_mug_seed*.cont.jsonl' {pull}/")
    theirs = {s for s in st if s not in mine}
    for f in pull.glob("hanging_mug_seed*.cont.jsonl"):
        seed = int(f.name[16:22])
        add = [r for r in cont_rows(f) if (seed, r["decision"]) in theirs and (seed, r["decision"]) not in tainted]
        if not add:
            continue
        dst = RD / f.name
        have = {(r["decision"], r["cand"]) for r in cont_rows(dst)}
        new = [r for r in add if (r["decision"], r["cand"]) not in have]
        if dst.exists():
            shutil.copy(dst, ARCH / f"{dst.name}.premerge")
        with dst.open("a") as fo:
            for r in new:
                fo.write(json.dumps(r) + "\n")
    log("cluster rows merged")
    # tainted states: archive rows + logs, rerun once on Amax with fresh logs
    if tainted:
        drop_states(RD.glob("hanging_mug_seed*.cont.jsonl"), set(tainted))
        for s, d in tainted:
            lg = RD / "gate_logs_Amax-Server" / f"seed{s}_d{d}.log"
            if lg.exists():
                lg.rename(ARCH / f"{lg.name}.tainted")
        (RD / "GATE_CONT_DONE_Amax-Server.json").rename(ARCH / "GATE_CONT_DONE_Amax-Server.retest_main.json")
        sh(f"cd {M}; setsid nohup python3 code/gate_run.py --machine amax --round-dir {RD} --gpus 4,5,6,7,8 "
           f"--per-gpu 1 --pairs-file {PAIRS} --states {','.join(f'{s}:{d}' for s, d in tainted)} "
           f"> /dev/null 2>&1 < /dev/null &")
        log(f"rerunning {len(tainted)} tainted states on Amax")
        while not (RD / "GATE_CONT_DONE_Amax-Server.json").exists():
            time.sleep(60)
        still = []
        for s, d in tainted:
            out = sh(f"grep -ciE '{BAD}' {RD}/gate_logs_Amax-Server/seed{s}_d{d}.log 2>/dev/null || true").split()
            if out and int(out[0]) > 0:
                still.append((s, d))
        if still:
            drop_states(RD.glob("hanging_mug_seed*.cont.jsonl"), set(still))
        log(f"rerun done; still tainted (excluded): {still}")
    # 5. evaluate
    allst = json.loads((RD / "gate_pairs.json").read_text())["states"] + json.loads((RD / PAIRS).read_text())["states"]
    (RD / "gate_pairs_pooled296.json").write_text(json.dumps(dict(note="descriptive only (PROTOCOL addendum 2.5)",
                                                                   states=allst)))
    sh(f"cd {M} && {PY} code/gate_eval.py --round-dir {RD} --scan-start 170000 --critic {RD}/critic_gate.json "
       f"--pairs-file {PAIRS} --out {RD}/gate_retest_result.json", 3600)
    sh(f"cd {M} && {PY} code/gate_eval.py --round-dir {RD} --scan-start 170000 --critic {RD}/critic_gate.json "
       f"--pairs-file gate_pairs_pooled296.json --out {RD}/gate_pooled296_result.json", 3600)
    r = json.loads((RD / "gate_retest_result.json").read_text())
    log(f"RESULT states {r['evaluated_states']} (incomplete {len(r['incomplete_states'])}) random {r['random']:.3f} "
        f"critic {r['picked']:.3f} gain {r['gain']:+.4f} CI {r['gain_ci']} pass {r['gate_pass']}")
    (RD / "GATE_RETEST_DONE").write_text(json.dumps(dict(gate_pass=r["gate_pass"], gain=r["gain"], ci=r["gain_ci"])) + "\n")


if __name__ == "__main__":
    main()
