"""Gate continuations for one round on one machine (Claude, 2026-11-22; PROTOCOL step 5, addendum 3-4). System python3.
Reads <round dir>/gate_pairs.json (gate_select.py) and runs this machine's share of the drawn states: state j (in the
file's order) belongs to this machine iff j % MOD is in REMS. For every listed card that is fully idle (no compute
process, < 1 GB used) it starts K standard probe servers (pi0, fast mode: --enable_prefix_cache
--skip_future_state_value; slot k on port 8300+10*smi+k), each with ONE driver thread (the FACT server serves one
connection at a time); the drivers pull states from one queue; a state is
one continue_candidates.py call with its 8 pairs d:0..d:7 (timeout 1 h). After the queue is empty, states with
missing or crashed pairs are retried once. Servers are stopped at the end; GATE_CONT_DONE_<host>.json summarizes.
Never touches other users' processes.
usage: setsid nohup python3 gate_run.py --machine amax|cluster --round-dir <dir> --gpus 0,1 --per-gpu 2 --mod 3 --rems 0 \
         > /dev/null 2>&1 < /dev/null &
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
    ap.add_argument("--gpus", required=True); ap.add_argument("--per-gpu", type=int, default=4)
    ap.add_argument("--mod", type=int, default=1); ap.add_argument("--rems", default="0")
    ap.add_argument("--pairs-file", default="gate_pairs.json", help="state list in the round dir")
    ap.add_argument("--states", default="", help="explicit seed:decision list (overrides mod/rems), e.g. reruns")
    ap.add_argument("--cc-args", default="", help="extra continue_candidates.py arguments, e.g. --fast-obs")
    a = ap.parse_args()
    c, rd, host = CFG[a.machine], a.round_dir, socket.gethostname()
    logf = rd / f"gate_run_{host}.log"
    lock = threading.Lock()

    def log(msg):
        with lock, logf.open("a") as f:
            f.write(time.strftime("%F %T ") + msg + "\n")

    gp = json.loads((rd / a.pairs_file).read_text())
    rems = {int(x) for x in a.rems.split(",")}
    states = [(s["seed"], s["decision"]) for j, s in enumerate(gp["states"]) if j % a.mod in rems]
    if a.states:
        want = {tuple(map(int, x.split(":"))) for x in a.states.split(",")}
        states = [st for st in [(s["seed"], s["decision"]) for s in gp["states"]] if st in want]
    log(f"start ({a.pairs_file}): {len(states)} of {len(gp['states'])} states (mod {a.mod} rems {sorted(rems)}), gpus {a.gpus}, "
        f"per gpu {a.per_gpu}, continue_candidates extra args: '{a.cc_args}'")
    # One driver per server: the FACT server (world_action_model.sockets) serves ONE connection at a time, so each
    # driver thread gets its own server ("slot"); slot k on card g listens on 8300 + 10 g + k.
    servers = {}
    for g in [int(x) for x in a.gpus.split(",")]:
        if not idle(g):
            log(f"gpu {g} not idle -> skipped"); continue
        for k in range(a.per_gpu):
            v, port = f"gate_{rd.name}_gpu{g}_s{k}", 8300 + 10 * g + k
            sh(f"{c['start']} {v} {g} {port} {FAST}")
            servers[(g, k)] = (Path(c["sdir"]) / f"server_{v}", port)
            log(f"server {v} started on gpu {g} port {port}")
            time.sleep(20)                     # stagger model loading on the same card
    ready = {}
    for slot, (d, port) in servers.items():
        for _ in range(120):
            if "ready and listening" in (d / "server.log").read_text(errors="ignore"):
                ready[slot] = port; break
            time.sleep(10)
        log(f"slot {slot} server {'ready' if slot in ready else 'NOT READY -> unused'}")
    if not ready:
        log("no server ready; abort"); return

    def done_pairs(seed):
        f = rd / f"hanging_mug_seed{seed}.cont.jsonl"
        if not f.exists():
            return set()
        return {(r["decision"], r["cand"]) for r in map(json.loads, f.read_text().splitlines()) if r}

    def run_state(slot, seed, d):
        g, port = slot[0], ready[slot]
        ep = rd / f"hanging_mug_seed{seed}.json"
        pairs = ",".join(f"{d}:{i}" for i in range(8))
        t0 = time.time()
        cmd = (f"FACT_ARCH=8.9 source {c['env']} {g} > /dev/null; export PYTHONPATH=$PYTHONPATH:{c['code']}; "
               f"exec timeout 3600 python {c['code']}/continue_candidates.py --episode {ep} --pairs {pairs} --port {port} "
               f"{a.cc_args}")
        with (rd / f"gate_logs_{host}" / f"seed{seed}_d{d}.log").open("a") as lf:
            rc = subprocess.run(["bash", "-c", cmd], stdout=lf, stderr=subprocess.STDOUT).returncode
        got = sum(1 for i in range(8) if (d, i) in done_pairs(seed))
        log(f"state {seed}:{d} slot {slot} rc {rc} pairs {got}/8 {time.time() - t0:.0f}s")
        return got == 8

    (rd / f"gate_logs_{host}").mkdir(exist_ok=True)
    q = queue.Queue()
    for st in states:
        q.put(st)
    failed = []

    def worker(slot):
        while True:
            try:
                seed, d = q.get_nowait()
            except queue.Empty:
                return
            if not run_state(slot, seed, d):
                with lock:
                    failed.append((seed, d))
    threads = [threading.Thread(target=worker, args=(slot,)) for slot in ready]
    for t in threads:
        t.start(); time.sleep(15)
    for t in threads:
        t.join()
    retry = list(failed); failed.clear()
    log(f"first pass done; retrying {len(retry)} states")
    for st in retry:
        q.put(st)
    threads = [threading.Thread(target=worker, args=(slot,)) for slot in ready]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for slot, (d, port) in servers.items():
        pid = (d / "pid").read_text().strip()
        sh(f"kill -- -$(ps -o pgid= -p {pid} | tr -d ' ') 2>/dev/null; sleep 5; kill -9 {pid} 2>/dev/null")
        log(f"server of slot {slot} stopped")
    summary = dict(host=host, states=len(states), incomplete=sorted(failed))
    (rd / f"GATE_CONT_DONE_{host}.json").write_text(json.dumps(summary) + "\n")
    log(f"done: {summary}")


if __name__ == "__main__":
    main()
