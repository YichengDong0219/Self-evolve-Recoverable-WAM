"""Offline-execution scheduler for one TTS round on Amax (Claude, 2026-11-21; protocol 12, step 1). System python3.
v2 (11-21 23:50, after the OOM incident in which v1 packed exec jobs next to the rollout servers): exec jobs never
share a card with rollouts or with the PRM service. A card (smi 4-8) is eligible only when
  - its rollout worker for this round has finished both passes ("second pass done" in progress_gpu<G>.txt),
  - it is not in --reserve (kept for the PRM service),
  - every compute process on it is ours (a card with any other user's process is never used),
  - it has < --max-used MiB in use and fewer than --per-gpu of our exec jobs.
For every finished episode (<ep>.json + <ep>.tts.npz) without <ep>.exec.json and without a running job, launch
exec_candidates.py (all candidates, no prefix rendering), longest episodes (most decisions) first.
Never touches other users' processes. Exits (touches EXEC_DONE) when every card's rollouts are done and every episode
has its exec output.
usage: setsid nohup python3 exec_scheduler.py --round-dir <dir> [--per-gpu 8] [--max-used 30000] [--reserve 4] \
         > /dev/null 2>&1 &
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

M = Path("/workspace/fact_local/mug_loop_1121")
GPUS = [4, 5, 6, 7, 8]
READY = "second pass done"


def used_mib(g):
    r = subprocess.run(["nvidia-smi", "-i", str(g), "--query-gpu=memory.used,name,uuid", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=20)
    used, name, uuid = r.stdout.strip().split(", ")
    return int(used), name, uuid


def only_ours(uuid):
    """True iff every compute process on this GPU belongs to us."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True)
    ap.add_argument("--per-gpu", type=int, default=8); ap.add_argument("--max-used", type=int, default=30000)
    ap.add_argument("--reserve", default="", help="comma-separated smi indices kept free (PRM service)")
    a = ap.parse_args()
    rd, log_path = a.round_dir, a.round_dir / "exec_scheduler.log"
    reserve = {int(x) for x in a.reserve.split(",") if x}
    gpus = [g for g in GPUS if g not in reserve]
    running, ndec, fails = {}, {}, {}   # name -> (Popen, gpu); name -> decisions; name -> failed attempts
    MAX_FAILS = 2

    def log(msg):
        with log_path.open("a") as f:
            f.write(time.strftime("%F %T ") + msg + "\n")

    def rollouts_done(g):
        p = rd / f"progress_gpu{g}.txt"
        return p.exists() and READY in p.read_text()

    def decisions(e):
        if e.name not in ndec:
            try:
                ndec[e.name] = (int(json.loads(e.read_text())["steps"]) + 47) // 48
            except Exception:  # noqa: BLE001
                ndec[e.name] = 0
        return ndec[e.name]

    log(f"start v2 per_gpu={a.per_gpu} max_used={a.max_used} reserve={sorted(reserve)} gpus={gpus}")
    while True:
        for name, (p, g) in list(running.items()):
            if p.poll() is not None:
                log(f"exec {name} gpu {g} rc {p.returncode}")
                del running[name]
                if not (rd / name).with_suffix(".exec.json").exists():
                    fails[name] = fails.get(name, 0) + 1
                    lg = (rd / name).with_suffix(".exec.log")
                    if lg.exists():
                        lg.rename(lg.with_name(lg.name + f".fail{fails[name]}"))
                    log(f"exec {name} FAILED (attempt {fails[name]}/{MAX_FAILS})")
        eps = [e for e in rd.glob("hanging_mug_seed*.json") if e.name.count(".") == 1]
        todo = [e for e in eps if e.with_suffix(".tts.npz").exists() and not e.with_suffix(".exec.json").exists()
                and e.name not in running and fails.get(e.name, 0) < MAX_FAILS]
        todo.sort(key=lambda e: (-decisions(e), e.name))
        for e in todo:
            cnt = {g: sum(1 for _, gg in running.values() if gg == g) for g in gpus}
            pick = None
            for g in sorted(gpus, key=lambda x: cnt[x]):
                if cnt[g] >= a.per_gpu or not rollouts_done(g):
                    continue
                used, name, uuid = used_mib(g)
                if "4090" in name and used < a.max_used and only_ours(uuid):
                    pick = g; break
            if pick is None:
                break
            cmd = (f"FACT_ARCH=8.9 source /workspace/fact_local/env.sh {pick}; export PYTHONPATH=$PYTHONPATH:{M}/code; "
                   f"exec python {M}/code/exec_candidates.py --episode {e} > {e.with_suffix('.exec.log')} 2>&1")
            p = subprocess.Popen(["bash", "-c", cmd], start_new_session=True)
            running[e.name] = (p, pick)
            log(f"launch {e.name} (D={decisions(e)}) on gpu {pick} pid {p.pid}")
            time.sleep(20)   # let the new job allocate its GPU memory before the next memory check
        if all(rollouts_done(g) for g in GPUS) and not running and not todo:
            bad = sorted(n for n, k in fails.items() if k >= MAX_FAILS)
            log(f"all done; episodes without exec output after {MAX_FAILS} attempts: {bad}")
            (rd / "EXEC_DONE").write_text(json.dumps(dict(failed=bad)) + "\n"); return
        time.sleep(60)


if __name__ == "__main__":
    main()
