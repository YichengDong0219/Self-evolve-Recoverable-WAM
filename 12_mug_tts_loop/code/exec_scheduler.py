"""Offline-execution scheduler for one TTS round on Amax (Claude, 2026-11-21; protocol 12, step 1). System python3.
Every 60 s: for each finished episode (<ep>.json + <ep>.tts.npz) without <ep>.exec.json and without a running job,
launch exec_candidates.py (all candidates, no prefix rendering) on an Amax 4090 (smi 4-8) whose compute processes are
ALL ours (our rollout servers / exec jobs; a card with any other user's process is never used), that has < MAX_USED
MiB in use and fewer than PER_GPU of our exec jobs. Never touches other users' processes. Exits when the round's workers are done and every episode has its exec output.
usage: setsid nohup python3 exec_scheduler.py <round dir> [per_gpu] [max_used_mib] > /dev/null 2>&1 &
"""
import json
import subprocess
import sys
import time
from pathlib import Path

RD = Path(sys.argv[1]); PER_GPU = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MAX_USED = int(sys.argv[3]) if len(sys.argv) > 3 else 44000
M = Path("/workspace/fact_local/mug_loop_1121"); LOG = RD / "exec_scheduler.log"
GPUS = [4, 5, 6, 7, 8]
running = {}   # episode name -> (Popen, gpu)


def log(msg):
    with LOG.open("a") as f:
        f.write(time.strftime("%F %T ") + msg + "\n")


def used_mib(g):
    r = subprocess.run(["nvidia-smi", "-i", str(g), "--query-gpu=memory.used,name,uuid", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=20)
    used, name, uuid = r.stdout.strip().split(", ")
    return int(used), name, uuid


def only_ours(uuid):
    """True iff every compute process on this GPU belongs to us (rollout servers / our exec jobs) -- never share a card
    with another user's job."""
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=20)
    for line in r.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2 or parts[0] != uuid:
            continue
        u = subprocess.run(["ps", "-o", "user=", "-p", parts[1]], capture_output=True, text=True).stdout.strip()
        if u != "$USER":
            return False
    return True


def workers_done():
    outs = list(RD.glob("progress_gpu*.txt"))
    return outs and all("done, server stopped" in p.read_text() for p in outs)


def main():
    log(f"start per_gpu={PER_GPU} max_used={MAX_USED}")
    while True:
        for name, (p, g) in list(running.items()):
            if p.poll() is not None:
                log(f"exec {name} gpu {g} rc {p.returncode}")
                del running[name]
        eps = sorted(e for e in RD.glob("hanging_mug_seed*.json") if e.name.count(".") == 1)
        todo = [e for e in eps if e.with_suffix(".tts.npz").exists() and not e.with_suffix(".exec.json").exists()
                and e.name not in running]
        for e in todo:
            cnt = {g: sum(1 for _, gg in running.values() if gg == g) for g in GPUS}
            pick = None
            for g in sorted(GPUS, key=lambda x: cnt[x]):
                used, name, uuid = used_mib(g)
                if "4090" in name and used < MAX_USED and cnt[g] < PER_GPU and only_ours(uuid):
                    pick = g; break
            if pick is None:
                break
            cmd = (f"FACT_ARCH=8.9 source /workspace/fact_local/env.sh {pick}; export PYTHONPATH=$PYTHONPATH:{M}/code; "
                   f"exec python {M}/code/exec_candidates.py --episode {e} > {e.with_suffix('.exec.log')} 2>&1")
            p = subprocess.Popen(["bash", "-c", cmd], start_new_session=True)
            running[e.name] = (p, pick)
            log(f"launch {e.name} on gpu {pick} pid {p.pid}")
        if workers_done() and not running and not todo:
            log("all done"); (RD / "EXEC_DONE").touch(); return
        time.sleep(60)


if __name__ == "__main__":
    main()
