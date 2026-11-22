"""Multi-chunk unit measurement runner for one machine (Claude, 2026-11-22; PROTOCOL addendum 3). System python3.
Same server/driver layout as gate_run.py (one standard probe server per fully idle card, pi0 fast mode, ONE driver
per server). Jobs = (state, unit K) for the states of <out-dir>/unit_states.json whose index j satisfies j % MOD in
REMS (all units of a state stay on one machine); a job is one continue_record.py call with the state's 8 pairs
(timeout 1 h). Jobs with missing pairs are retried once. Writes <out-dir>/unit_run_<host>.log, per-job logs in
<out-dir>/logs_<host>/ and <out-dir>/UNIT_DONE_<host>.json. Never touches other users' processes.
usage: setsid nohup python3 unit_run.py --machine amax|cluster --round-dir <r0> --out-dir <r0>/unit_measure \
         --gpus 4,5 --mod 9 --rems 0,1,2,3,4 [--units 1,2,3] [--jobs s:d:K,...] > /dev/null 2>&1 < /dev/null &
"""
import argparse
import json
import queue
import socket
import subprocess
import threading
import time
from pathlib import Path

CFG = {
    "amax": dict(env="/workspace/fact_local/env.sh", code="/workspace/fact_local/mug_loop_1121/code",
                 start="bash /workspace/fact_local/code/start_server_amax.sh",
                 sdir="/workspace/fact_local/runs_server"),
    "cluster": dict(env="/workspace_cluster/fact_local/env.sh", code="/workspace_cluster/fact_local/mug_loop_1121/code",
                    start="bash /workspace_cluster/fact_local/mug_loop_1121/code/start_server_probe_cluster.sh",
                    sdir="/workspace_cluster/fact_local/mug_loop_1121/runs_server"),
}
FAST = "--enable_prefix_cache --skip_future_state_value"


def sh(cmd, timeout=30):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()


def idle(g):
    apps = sh(f"nvidia-smi -i {g} --query-compute-apps=pid --format=csv,noheader")
    used = int(sh(f"nvidia-smi -i {g} --query-gpu=memory.used --format=csv,noheader,nounits"))
    return apps == "" and used < 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", choices=list(CFG), required=True); ap.add_argument("--round-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--gpus", required=True)
    ap.add_argument("--mod", type=int, default=1); ap.add_argument("--rems", default="0")
    ap.add_argument("--units", default="1,2,3"); ap.add_argument("--jobs", default="", help="explicit s:d:K list (reruns)")
    a = ap.parse_args()
    c, rd, od, host = CFG[a.machine], a.round_dir, a.out_dir, socket.gethostname()
    od.mkdir(parents=True, exist_ok=True); (od / f"logs_{host}").mkdir(exist_ok=True)
    lock = threading.Lock()

    def log(msg):
        with lock, (od / f"unit_run_{host}.log").open("a") as f:
            f.write(time.strftime("%F %T ") + msg + "\n")

    st = [(s["seed"], s["decision"]) for s in json.loads((od / "unit_states.json").read_text())["states"]]
    rems = {int(x) for x in a.rems.split(",")}
    units = [int(x) for x in a.units.split(",")]
    jobs = [(s, d, K) for j, (s, d) in enumerate(st) if j % a.mod in rems for K in units]
    if a.jobs:
        jobs = [tuple(map(int, x.split(":"))) for x in a.jobs.split(",")]
    log(f"start: {len(jobs)} jobs (units {units}, mod {a.mod} rems {sorted(rems)}), gpus {a.gpus}")
    servers = {}
    for g in [int(x) for x in a.gpus.split(",")]:
        if not idle(g):
            log(f"gpu {g} not idle -> skipped"); continue
        v, port = f"unit_{rd.name}_gpu{g}_s0", 8300 + 10 * g
        sh(f"{c['start']} {v} {g} {port} {FAST}")
        servers[g] = (Path(c["sdir"]) / f"server_{v}", port)
        log(f"server {v} started on gpu {g} port {port}")
        time.sleep(20)
    ready = {}
    for g, (d, port) in servers.items():
        for _ in range(120):
            if "ready and listening" in (d / "server.log").read_text(errors="ignore"):
                ready[g] = port; break
            time.sleep(10)
        log(f"gpu {g} server {'ready' if g in ready else 'NOT READY -> unused'}")
    if not ready:
        log("no server ready; abort"); return

    def n_done(seed, d, K):
        f = od / f"seed{seed}.unit{K}.jsonl"
        if not f.exists():
            return 0
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        return len({r["cand"] for r in rows if r["decision"] == d and r.get("start_ok") and "success" in r})

    def run_job(g, seed, d, K):
        ep = rd / f"hanging_mug_seed{seed}.json"
        pairs = ",".join(f"{d}:{i}" for i in range(8))
        t0 = time.time()
        cmd = (f"FACT_ARCH=8.9 source {c['env']} {g} > /dev/null; export PYTHONPATH=$PYTHONPATH:{c['code']}; "
               f"exec timeout 3600 python {c['code']}/continue_record.py --episode {ep} --pairs {pairs} "
               f"--port {ready[g]} --unit {K} --out-dir {od}")
        with (od / f"logs_{host}" / f"seed{seed}_d{d}_u{K}.log").open("a") as lf:
            rc = subprocess.run(["bash", "-c", cmd], stdout=lf, stderr=subprocess.STDOUT).returncode
        got = n_done(seed, d, K)
        log(f"job {seed}:{d}:K{K} gpu {g} rc {rc} pairs {got}/8 {time.time() - t0:.0f}s")
        return got == 8

    q = queue.Queue()
    for jb in jobs:
        q.put(jb)
    failed = []

    def worker(g):
        while True:
            try:
                jb = q.get_nowait()
            except queue.Empty:
                return
            if not run_job(g, *jb):
                with lock:
                    failed.append(jb)
    for rnd in range(2):
        threads = [threading.Thread(target=worker, args=(g,)) for g in ready]
        for t in threads:
            t.start(); time.sleep(15)
        for t in threads:
            t.join()
        if rnd == 0:
            retry = list(failed); failed.clear()
            log(f"first pass done; retrying {len(retry)} jobs")
            for jb in retry:
                q.put(jb)
    for g, (d, port) in servers.items():
        pid = (d / "pid").read_text().strip()
        sh(f"kill -- -$(ps -o pgid= -p {pid} | tr -d ' ') 2>/dev/null; sleep 5; kill -9 {pid} 2>/dev/null")
        log(f"server on gpu {g} stopped")
    summary = dict(host=host, jobs=len(jobs), incomplete=sorted(failed))
    (od / f"UNIT_DONE_{host}.json").write_text(json.dumps(summary) + "\n")
    log(f"done: {summary}")


if __name__ == "__main__":
    main()
