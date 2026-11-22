"""Unit measurement runner with the FACT probe servers on the Amax RTX PRO 6000s (Claude, 2026-11-22; user: "能否在
Pro6000上也并行"). System python3. The servers run in the fact_bw venv (torch 2.7.1+cu128, sm_120) via
start_server_bw.sh with the same flags as the 4090 servers; the drivers (RoboTwin sim + cuRobo, whose torch has no
sm_120 kernels) run on the Amax 4090s next to our own jobs, at most 2 extra drivers per 4090. Jobs = (state, K) for
the states in <out-dir>/unit_reassign.json "to_bw", K = 1, 2, 3, processed in REVERSE queue order (the states right
after the 4090 runners' kept ones come last, so a hand-off race on them is settled long before). Same job command,
retry and output as unit_run.py; names tagged "<host>-bw": unit_run_<host>-bw.log, logs_<host>-bw/,
UNIT_DONE_<host>-bw.json. Never touches other users' processes.
usage: setsid nohup python3 code/unit_run_bw.py --round-dir r0 --out-dir r0/unit_measure --server-gpus 0,1,2,3 \
         --per-gpu 2 --driver-gpus 4,5,6,7,8 > /dev/null 2>&1 < /dev/null &
"""
import argparse
import json
import os
import queue
import socket
import subprocess
import threading
import time
from pathlib import Path

ENV = "/workspace/fact_local/env.sh"
CODE = "/workspace/fact_local/mug_loop_1121/code"
START = f"bash {CODE}/start_server_bw.sh"
SDIR = "/workspace/fact_local/runs_server"
FAST = "--enable_prefix_cache --skip_future_state_value"


def sh(cmd, timeout=30):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()


def apps(g):
    return [int(x) for x in sh(f"nvidia-smi -i {g} --query-compute-apps=pid --format=csv,noheader").split() if x.isdigit()]


def used(g):
    return int(sh(f"nvidia-smi -i {g} --query-gpu=memory.used --format=csv,noheader,nounits"))


def mine(pid):
    try:
        return os.stat(f"/proc/{pid}").st_uid == os.getuid()
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--server-gpus", default="0,1,2,3"); ap.add_argument("--per-gpu", type=int, default=2)
    ap.add_argument("--driver-gpus", default="4,5,6,7,8"); ap.add_argument("--max-extra", type=int, default=2)
    a = ap.parse_args()
    rd, od = a.round_dir, a.out_dir
    host = socket.gethostname() + "-bw"
    (od / f"logs_{host}").mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def log(msg):
        with lock, (od / f"unit_run_{host}.log").open("a") as f:
            f.write(time.strftime("%F %T ") + msg + "\n")

    ra = json.loads((od / "unit_reassign.json").read_text())
    jobs = [(s, d, K) for s, d in reversed([tuple(x) for x in ra["to_bw"]]) for K in (1, 2, 3)]
    log(f"start: {len(jobs)} jobs ({len(ra['to_bw'])} states, reverse queue order), server gpus {a.server_gpus} x "
        f"{a.per_gpu}, driver gpus {a.driver_gpus}")
    dgs = []
    for g in [int(x) for x in a.driver_gpus.split(",")]:   # driver cards: only our own processes, memory headroom
        if not all(mine(p) for p in apps(g)) or used(g) > 32000:
            log(f"driver gpu {g} has a foreign process or > 32 GB used -> not used for drivers")
        else:
            dgs.append(g)
    servers, extra = {}, {g: 0 for g in dgs}
    for g in [int(x) for x in a.server_gpus.split(",")]:
        if apps(g) or used(g) >= 1024:
            log(f"server gpu {g} not idle -> skipped"); continue
        for k in range(a.per_gpu):
            cand = [x for x in dgs if extra[x] < a.max_extra]
            if not cand:
                log("no driver card with headroom left"); break
            dg = min(cand, key=lambda x: (extra[x], x)); extra[dg] += 1
            v, port = f"unitbw_{rd.name}_gpu{g}_s{k}", 8400 + 10 * g + k
            sh(f"{START} {v} {g} {port} {FAST}")
            servers[(g, k)] = (Path(SDIR) / f"server_{v}", port, dg)
            log(f"server {v} started on gpu {g} port {port}; its driver runs on 4090 gpu {dg}")
            time.sleep(20)
    ready = {}
    for slot, (d, port, dg) in servers.items():
        for _ in range(120):
            if "ready and listening" in (d / "server.log").read_text(errors="ignore"):
                ready[slot] = (port, dg); break
            time.sleep(10)
        log(f"slot {slot} server {'ready' if slot in ready else 'NOT READY -> unused'}")
    if not ready:
        log("no server ready; abort"); return

    def n_done(seed, d, K):
        f = od / f"seed{seed}.unit{K}.jsonl"
        if not f.exists():
            return 0
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        return len({r["cand"] for r in rows if r["decision"] == d and r.get("start_ok") and "success" in r})

    def run_job(slot, seed, d, K):
        port, dg = ready[slot]
        ep = rd / f"hanging_mug_seed{seed}.json"
        pairs = ",".join(f"{d}:{i}" for i in range(8))
        t0 = time.time()
        cmd = (f"FACT_ARCH=8.9 source {ENV} {dg} > /dev/null; export PYTHONPATH=$PYTHONPATH:{CODE}; "
               f"exec timeout 3600 python {CODE}/continue_record.py --episode {ep} --pairs {pairs} "
               f"--port {port} --unit {K} --out-dir {od}")
        with (od / f"logs_{host}" / f"seed{seed}_d{d}_u{K}.log").open("a") as lf:
            rc = subprocess.run(["bash", "-c", cmd], stdout=lf, stderr=subprocess.STDOUT).returncode
        got = n_done(seed, d, K)
        log(f"job {seed}:{d}:K{K} slot {slot} (driver gpu {dg}) rc {rc} pairs {got}/8 {time.time() - t0:.0f}s")
        return got == 8

    q = queue.Queue()
    for jb in jobs:
        q.put(jb)
    failed = []

    def worker(slot):
        while True:
            try:
                jb = q.get_nowait()
            except queue.Empty:
                return
            if not run_job(slot, *jb):
                with lock:
                    failed.append(jb)
    for rnd in range(2):
        threads = [threading.Thread(target=worker, args=(slot,)) for slot in ready]
        for t in threads:
            t.start(); time.sleep(15)
        for t in threads:
            t.join()
        if rnd == 0:
            retry = list(failed); failed.clear()
            log(f"first pass done; retrying {len(retry)} jobs")
            for jb in retry:
                q.put(jb)
    for slot, (d, port, dg) in servers.items():
        pid = (d / "pid").read_text().strip()
        sh(f"kill -- -$(ps -o pgid= -p {pid} | tr -d ' ') 2>/dev/null; sleep 5; kill -9 {pid} 2>/dev/null")
        log(f"server of slot {slot} stopped")
    summary = dict(host=host, jobs=len(jobs), incomplete=sorted(failed))
    (od / f"UNIT_DONE_{host}.json").write_text(json.dumps(summary) + "\n")
    log(f"done: {summary}")


if __name__ == "__main__":
    main()
