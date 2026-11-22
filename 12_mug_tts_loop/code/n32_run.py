"""Continuation runner for the N = 32 snapshot test on one machine (Claude, 2026-11-22; PROTOCOL addendum 4).
System python3. One multi-connection probe server (inference_server_multi.py: pi0 fast mode, the same flags as the
gate servers, CPU threads capped, one thread per connection, inference serialized) per fully idle card in --gpus, and
--per-server drivers on that card. Jobs = <out-dir>/n32_jobs.json [machine]: [seed, decision, [8 candidate ids]];
a job is one continue_n32.py call (timeout 1 h); jobs with missing pairs are retried once. Writes
<out-dir>/n32_run_<host>.log, per-job logs in <out-dir>/logs_<host>/ and <out-dir>/N32_DONE_<host>.json.
Never touches other users' processes.
usage: setsid nohup python3 n32_run.py --machine amax|cluster --round-dir <r0> --out-dir <dir> --gpus 5,6,7,8 \
         [--per-server 3] > /dev/null 2>&1 < /dev/null &
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
                 start="bash /workspace/fact_local/mug_loop_1121/code/start_server_multi.sh {v} {g} {p} fact",
                 sdir="/workspace/fact_local/runs_server"),
    "cluster": dict(env="/workspace_cluster/fact_local/env.sh", code="/workspace_cluster/fact_local/mug_loop_1121/code",
                    start="bash /workspace_cluster/fact_local/mug_loop_1121/code/start_server_multi_cluster.sh {v} {g} {p}",
                    sdir="/workspace_cluster/fact_local/mug_loop_1121/runs_server"),
}
FAST = "--enable_prefix_cache --skip_future_state_value"


def sh(cmd, timeout=30):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()


def idle(g):
    apps = sh(f"nvidia-smi -i {g} --query-compute-apps=pid --format=csv,noheader")
    used = sh(f"nvidia-smi -i {g} --query-gpu=memory.used --format=csv,noheader,nounits")
    return apps == "" and used.isdigit() and int(used) < 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", choices=list(CFG), required=True); ap.add_argument("--round-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--gpus", required=True)
    ap.add_argument("--per-server", type=int, default=3)
    a = ap.parse_args()
    c, rd, od, host = CFG[a.machine], a.round_dir, a.out_dir, socket.gethostname()
    (od / f"logs_{host}").mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def log(msg):
        with lock, (od / f"n32_run_{host}.log").open("a") as f:
            f.write(time.strftime("%F %T ") + msg + "\n")

    jobs = [(s, d, tuple(ids)) for s, d, ids in json.loads((od / "n32_jobs.json").read_text())[a.machine]]
    log(f"start: {len(jobs)} jobs ({sum(len(j[2]) for j in jobs)} pairs), gpus {a.gpus}, {a.per_server} drivers per server")
    servers = {}
    for g in [int(x) for x in a.gpus.split(",")]:
        if not idle(g):
            log(f"gpu {g} not idle -> skipped"); continue
        v, port = f"n32_{a.machine}_gpu{g}", 8800 + g
        sh(c["start"].format(v=v, g=g, p=port) + " " + FAST)
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

    def n_done(seed, d, ids):
        f = od / f"seed{seed}.n32.jsonl"
        if not f.exists():
            return 0
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        return len({r["cand"] for r in rows if r["decision"] == d and r["cand"] in ids and r.get("start_ok") and "success" in r})

    def run_job(g, seed, d, ids):
        ep = rd / f"hanging_mug_seed{seed}.json"
        pairs = ",".join(f"{d}:{i}" for i in ids)
        t0 = time.time()
        cmd = (f"FACT_ARCH=8.9 source {c['env']} {g} > /dev/null; export PYTHONPATH=$PYTHONPATH:{c['code']}; "
               f"exec timeout 3600 python {c['code']}/continue_n32.py --episode {ep} --pairs {pairs} "
               f"--port {ready[g]} --cands-dir {od}/cands --out-dir {od}")
        with (od / f"logs_{host}" / f"seed{seed}_d{d}_c{ids[0]}.log").open("a") as lf:
            rc = subprocess.run(["bash", "-c", cmd], stdout=lf, stderr=subprocess.STDOUT).returncode
        got = n_done(seed, d, ids)
        log(f"job {seed}:{d}:c{ids[0]}-{ids[-1]} gpu {g} rc {rc} pairs {got}/{len(ids)} {time.time() - t0:.0f}s")
        return got == len(ids)

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
    slots = [g for g in ready for _ in range(a.per_server)]
    for rnd in range(2):
        threads = [threading.Thread(target=worker, args=(g,)) for g in slots]
        for t in threads:
            t.start(); time.sleep(10)
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
    summary = dict(host=host, jobs=len(jobs), incomplete=[[s, d, list(ids)] for s, d, ids in sorted(failed)])
    (od / f"N32_DONE_{host}.json").write_text(json.dumps(summary) + "\n")
    log(f"done: {summary}")


if __name__ == "__main__":
    main()
