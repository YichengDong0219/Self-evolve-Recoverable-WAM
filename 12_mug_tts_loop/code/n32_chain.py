"""N = 32 snapshot test end to end, on Amax (Claude, 2026-11-22; PROTOCOL addendum 4; user: "选部分轨迹，在每个
snapshot出发rollout32次，然后让现有的critic打分，看看能否得到比K=8更高的多样性"). System python3.
 1. generation: TTS servers (start_server_tts_amax.sh) on the fully idle Amax cards among 4-8, one n32_generate.py
    driver per server, one scene (decisions 0-6) per job, retried once; then the TTS servers are stopped;
 2. audit: start_ok, candidate ids 0-7 equal to round 0, render errors in the logs (such states are excluded and
    reported, not rerun);
 3. jobs: per state ids 8-31 if its ids 0-7 reproduce round 0 (their gate outcomes are reused), else ids 0-31; blocks
    of 8; states (sorted) with index j % 9 in 0..4 -> Amax, else -> cluster; written to n32_jobs.json;
 4. sync to the cluster (code sha-checked, jobs, candidate files, missing episode files) and launch n32_run.py there
    (cards 0-3) and on Amax (cards 5-8), 3 drivers per multi-connection server;
 5. critic scores: prm_service.py with the Q part of r0/critic_gate.json on Amax card 4 (n32_score.py), then stop it;
 6. wait for both N32_DONE; audit the job logs; pull the cluster rows; n32_analysis.py -> n32_result.txt; N32_ALL_DONE.
Never touches other users' processes. Log: r0/n32/n32_chain.log.
usage: setsid nohup python3 code/n32_chain.py [--from-step N] > /dev/null 2>&1 < /dev/null &
"""
import argparse
import hashlib
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

M = Path("/workspace/fact_local/mug_loop_1121"); RD = M / "r0"; OD = RD / "n32"; GEN = OD / "gen"
CL = "$USER@CLUSTER_HOST"; CM = "/workspace_cluster/fact_local/mug_loop_1121"; CRD = CM + "/r0"; COD = CRD + "/n32"
ENV = "/workspace/fact_local/env.sh"
PY = "/workspace/miniconda3/envs/RoboTwin/bin/python"
PRMPY = "/workspace/venvs/prm_dopamine/bin/python"; P = "/workspace/prm"
CL_CODE = ["continue_n32.py", "n32_run.py", "start_server_multi_cluster.sh", "inference_server_multi.py",
           "continue_record.py", "exec_candidates.py", "unit_common.py", "prm_common.py"]
BAD = r"OIDN|DeviceLost|vk::|cannot create|out of memory"
DECISIONS = list(range(7))
LOG = OD / "n32_chain.log"


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


def stop_pid_file(pid_file):
    pid = Path(pid_file).read_text().strip()
    sh(f"kill -- -$(ps -o pgid= -p {pid} | tr -d ' ') 2>/dev/null; sleep 5; kill -9 {pid} 2>/dev/null; true", 60)


def gen_rows(seed):
    f = GEN / f"gen_seed{seed}.jsonl"
    return {r["decision"]: r for r in map(json.loads, f.read_text().splitlines()) if r} if f.exists() else {}


def generation(scenes):
    cards = [g for g in (4, 5, 6, 7, 8) if idle(g)]
    log(f"generation: idle cards {cards}")
    if not cards:
        raise SystemExit("no idle Amax card for generation")
    srv = {}
    for g in cards:
        v = f"n32tts_gpu{g}"
        sh(f"bash {M}/code/start_server_tts_amax.sh {v} {g} {8700 + g}", 60)
        srv[g] = M / "runs_server" / f"server_{v}"
        time.sleep(20)
    ready = []
    for g, d in srv.items():
        for _ in range(120):
            if "ready and listening" in (d / "server.log").read_text(errors="ignore"):
                ready.append(g); break
            time.sleep(10)
        log(f"TTS server gpu {g} {'ready' if g in ready else 'NOT READY'}")
    (GEN / "logs").mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def complete(s):
        r = gen_rows(s)
        return all(d in r and r[d].get("start_ok") and r[d].get("n") == 32 for d in DECISIONS)

    def worker(g, q):
        while True:
            try:
                s = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            cmd = (f"FACT_ARCH=8.9 source {ENV} {g} > /dev/null; export PYTHONPATH=$PYTHONPATH:{M}/code; "
                   f"exec timeout 7200 python {M}/code/n32_generate.py --episode {RD}/hanging_mug_seed{s}.json "
                   f"--decisions {','.join(map(str, DECISIONS))} --port {8700 + g} --out-dir {GEN}")
            with (GEN / "logs" / f"gen_seed{s}.log").open("a") as lf:
                rc = subprocess.run(["bash", "-c", cmd], stdout=lf, stderr=subprocess.STDOUT).returncode
            with lock:
                log(f"generation scene {s} gpu {g} rc {rc} complete {complete(s)} {time.time() - t0:.0f}s")
    for rnd in range(2):
        q = queue.Queue()
        for s in scenes:
            if not complete(s):
                q.put(s)
        if q.empty():
            break
        if rnd:
            log(f"generation retry: {q.qsize()} scenes")
        th = [threading.Thread(target=worker, args=(g, q)) for g in ready]
        for t in th:
            t.start(); time.sleep(5)
        for t in th:
            t.join()
    for g, d in srv.items():
        stop_pid_file(d / "pid")
    log("TTS servers stopped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-step", type=int, default=1)
    a = ap.parse_args()
    OD.mkdir(parents=True, exist_ok=True); GEN.mkdir(exist_ok=True); (OD / "cands").mkdir(exist_ok=True)
    gate = json.loads((RD / "critic_gate.json").read_text())
    scenes = sorted(gate["held_out"])
    states = [(s, d) for s in scenes for d in DECISIONS]
    st_json = json.dumps(dict(scenes=scenes, decisions=DECISIONS, states=states))
    (OD / "n32_states.json").write_text(st_json)
    log(f"start (from step {a.from_step}): {len(states)} states, sha {hashlib.sha256(st_json.encode()).hexdigest()[:16]}")
    if a.from_step <= 1:
        generation(scenes)
    # 2. audit
    excl, use_gate = [], {}
    for s in scenes:
        rows = gen_rows(s)
        bad = sh(f"grep -ciE '{BAD}' {GEN}/logs/gen_seed{s}.log 2>/dev/null || true")
        tainted = bad and bad.split()[0].isdigit() and int(bad.split()[0]) > 0
        for d in DECISIONS:
            r = rows.get(d)
            if tainted or r is None or not r.get("start_ok") or r.get("n") != 32:
                excl.append([s, d, "render error" if tainted else "generation incomplete"]); continue
            use_gate[(s, d)] = bool(r["ids07_equal"])
    n_eq = sum(use_gate.values())
    log(f"generation audit: {len(use_gate)} states usable, {len(excl)} excluded {excl}; ids 0-7 equal to round 0 in "
        f"{n_eq}/{len(use_gate)} states")
    (OD / "gen_excluded.json").write_text(json.dumps(excl))
    # 3. jobs
    jobs = {"amax": [], "cluster": []}
    for j, (s, d) in enumerate(sorted(use_gate)):
        ids = list(range(8, 32)) if use_gate[(s, d)] else list(range(32))
        side = "amax" if j % 9 < 5 else "cluster"
        for b in range(0, len(ids), 8):
            jobs[side].append([s, d, ids[b:b + 8]])
        sh(f"cp {GEN}/seed{s}_d{d}.cands.npz {OD}/cands/")
    (OD / "n32_jobs.json").write_text(json.dumps(jobs))
    log(f"jobs: amax {len(jobs['amax'])}, cluster {len(jobs['cluster'])}")
    if a.from_step <= 4:
        # 4. sync + launch
        ssh(f"mkdir -p {COD}/cands")
        sh(f"scp -q {' '.join(str(M / 'code' / f) for f in CL_CODE)} {CL}:{CM}/code/")
        files = [str(M / "code" / f) for f in CL_CODE]
        a1 = sh("sha256sum " + " ".join(files) + " | cut -c1-64")
        b1 = ssh("sha256sum " + " ".join(f.replace(str(M), CM) for f in files) + " | cut -c1-64")
        if not a1 or a1 != b1:
            log(f"code sha mismatch between machines; abort\n{a1}\n{b1}"); raise SystemExit(1)
        cl_states = sorted({(s, d) for s, d, _ in jobs["cluster"]})
        sh(f"scp -q {' '.join(f'{OD}/cands/seed{s}_d{d}.cands.npz' for s, d in cl_states)} {CL}:{COD}/cands/", 1800)
        for s in sorted({s for s, _ in cl_states}):
            if ssh(f"ls {CRD}/hanging_mug_seed{s}.json 2>/dev/null") == "":
                sh(f"scp -q {RD}/hanging_mug_seed{s}.json {CL}:{CRD}/", 600); log(f"copied episode {s} to the cluster")
        sh(f"scp -q {OD}/n32_jobs.json {CL}:{COD}/")
        n_cl = ssh(f"ls {COD}/cands | wc -l")
        log(f"cluster synced: code identical ({len(files)} files), {n_cl} candidate files, jobs")
        ssh(f"cd {CM}; setsid nohup python3 code/n32_run.py --machine cluster --round-dir {CRD} --out-dir {COD} "
            f"--gpus 0,1,2,3 --per-server 3 > /dev/null 2>&1 < /dev/null &", 60)
        sh(f"cd {M}; setsid nohup python3 code/n32_run.py --machine amax --round-dir {RD} --out-dir {OD} "
           f"--gpus 5,6,7,8 --per-server 3 > /dev/null 2>&1 < /dev/null &", 60)
        log("n32_run launched on the cluster (0-3) and Amax (5-8)")
    if a.from_step <= 5:
        # 5. critic scores
        q = gate["Q"]
        (OD / "critic_gate_Q.json").write_text(json.dumps(q))
        log(f"critic: Q of critic_gate.json (sha256 {hashlib.sha256((RD / 'critic_gate.json').read_bytes()).hexdigest()[:16]}), "
            f"features {q['features']}, l2 {q['l2']}")
        g = None                       # card 4 only: 5-8 are being taken by the Amax runner's servers right now
        for _ in range(60):
            if idle(4):
                g = 4; break
            time.sleep(60)
        if g is None:
            log("no idle Amax card for the PRM critic service; scores missing")
        else:
            svc = (f"cd {M}/code; HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 "
                   f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={g} PYTHONUNBUFFERED=1 "
                   f"setsid {PRMPY} prm_service.py --model {P}/models/GRM-2.0-8B-Preview --port 8650 "
                   f"--critic {OD}/critic_gate_Q.json "
                   f"--prompt-src {P}/Robo-Dopamine/examples/inference.py --blank-goal {P}/Robo-Dopamine/examples/blank_goal.png "
                   f"> {OD}/prm_service.log 2>&1 < /dev/null & echo $!")
            spid = sh(svc, 60).split()[-1]
            log(f"PRM critic service pid {spid} on gpu {g}")
            for _ in range(90):
                if "PRM_SERVICE_READY" in (OD / "prm_service.log").read_text(errors="ignore"):
                    break
                time.sleep(20)
            if "PRM_SERVICE_READY" in (OD / "prm_service.log").read_text(errors="ignore"):
                out = sh(f"cd {M}/code; {PRMPY} n32_score.py --gen-dir {GEN} > {OD}/n32_score.out 2>&1; tail -1 {OD}/n32_score.out", 4 * 3600)
                log(f"critic scores: {out}")
            else:
                log("PRM critic service not ready after 30 min; scores missing")
            pg = sh(f"ps -o pgid= -p {spid} | tr -d ' '")
            if pg:
                sh(f"kill -- -{pg}; sleep 10; kill -9 -- -{pg} 2>/dev/null; true", 60)
            log("PRM critic service stopped")
    # 6. wait, audit, merge, analyse
    t0 = time.time()
    while True:
        a_done = (OD / "N32_DONE_Amax-Server.json").exists()
        c_done = ssh(f"ls {COD}/N32_DONE_user.json 2>/dev/null") != ""
        if a_done and c_done:
            break
        if time.time() - t0 > 10 * 3600:
            log("timeout waiting for n32_run (10 h); abort"); return
        time.sleep(120)
    log("runners done: Amax " + (OD / "N32_DONE_Amax-Server.json").read_text().strip() + " | cluster "
        + ssh(f"cat {COD}/N32_DONE_user.json"))
    tainted = []
    for side, base in (("amax", f"{OD}/logs_Amax-Server"), ("cluster", f"{COD}/logs_user")):
        for s, d, ids in jobs[side]:
            lg = f"{base}/seed{s}_d{d}_c{ids[0]}.log"
            n = (sh(f"grep -ciE '{BAD}' {lg} 2>/dev/null || true") if side == "amax"
                 else ssh(f"grep -ciE \"{BAD}\" {lg} 2>/dev/null || true"))
            if n and n.split()[0].isdigit() and int(n.split()[0]) > 0:
                tainted.append([s, d, ids])
    (OD / "n32_excluded_jobs.json").write_text(json.dumps(tainted))
    log(f"continuation audit: render-error jobs (excluded) {tainted}")
    pull = OD / "_cluster"; pull.mkdir(exist_ok=True)
    sh(f"scp -q '{CL}:{COD}/seed*.n32.jsonl' {pull}/", 1800)
    n_rows = 0
    for f in pull.glob("seed*.n32.jsonl"):
        dst = OD / f.name
        have = set()
        if dst.exists():
            have = {(r["decision"], r["cand"]) for r in map(json.loads, dst.read_text().splitlines()) if r}
        with dst.open("a") as fo:
            for line in f.read_text().splitlines():
                if line.strip() and (json.loads(line)["decision"], json.loads(line)["cand"]) not in have:
                    fo.write(line + "\n"); n_rows += 1
    log(f"cluster merged: {n_rows} rows")
    out = sh(f"cd {M}/code; {PY} n32_analysis.py --round-dir {RD} --out-dir {OD} > {OD}/n32_analysis.out 2>&1; "
             f"tail -4 {OD}/n32_analysis.out", 3600)
    log(f"analysis: {out}")
    (OD / "N32_ALL_DONE").write_text(time.strftime("%F %T") + "\n")
    log("done")


if __name__ == "__main__":
    main()
