"""Hand the tail of a 4090 unit_run.py queue to the PRO 6000 runner (Claude, 2026-11-22; user: "能否在Pro6000上也并行").
Runs on the machine of that runner, system python3. Reads <out-dir>/unit_reassign.json: "<side>_keep" = states this
runner still runs, "<side>_moved" = its states moved to unit_run_bw.py (fixed by queue position before any outcome is
looked at; every state keeps all its units K = 1, 2, 3 on one runner).
 1. every 3 s: a driver of this runner working on a moved state (dispatch race) is killed by its process group, and
    its log / rows / recordings of that job are archived at once (the PRO 6000 runner does such states last);
 2. once every kept job has been dispatched (its log exists), SIGTERM the runner (running drivers are separate
    processes and finish normally);
 3. when no driver of a kept job is left, stop the runner's servers by PID and write UNIT_DONE_<host>.json.
Only touches our own processes in the runner's session and the listed servers.
usage: setsid nohup python3 code/unit_handoff.py --out-dir <dir> --side amax|cluster --host Amax-Server|user \
         --runner-pid P --server-dirs '<glob>' > /dev/null 2>&1 < /dev/null &
"""
import argparse
import glob
import json
import os
import shutil
import signal
import time
from pathlib import Path


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def drivers(sid):
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            a = [x.decode() for x in (p / "cmdline").read_bytes().split(b"\0") if x]
            stat = (p / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(stat[3]) != sid or not any(x.endswith("continue_record.py") for x in a) or "timeout" in a[0]:
            continue
        seed = int(Path(a[a.index("--episode") + 1]).name[16:22])
        d = int(a[a.index("--pairs") + 1].split(":")[0]); K = int(a[a.index("--unit") + 1])
        out.append((int(p.name), int(stat[2]), seed, d, K))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--side", required=True)
    ap.add_argument("--host", required=True); ap.add_argument("--runner-pid", type=int, required=True)
    ap.add_argument("--server-dirs", required=True)
    a = ap.parse_args()
    od, host, rp = a.out_dir, a.host, a.runner_pid
    arch = od / "_handoff_archive"; arch.mkdir(exist_ok=True)
    logf = od / f"unit_handoff_{host}.log"

    def log(m):
        with logf.open("a") as f:
            f.write(time.strftime("%F %T ") + m + "\n")

    cfg = json.loads((od / "unit_reassign.json").read_text())
    keep = [tuple(x) for x in cfg[f"{a.side}_keep"]]; moved = {tuple(x) for x in cfg[f"{a.side}_moved"]}
    kept_jobs = [(s, d, K) for s, d in keep for K in (1, 2, 3)]
    sdirs = sorted(glob.glob(a.server_dirs))
    log(f"start: keep {len(keep)} states ({len(kept_jobs)} jobs), moved {len(moved)} states, runner {rp}, servers {sdirs}")

    def archive_job(s, d, K):
        lg = od / f"logs_{host}" / f"seed{s}_d{d}_u{K}.log"
        if lg.exists():
            lg.rename(arch / f"{lg.name}.{host}.moved")
        f = od / f"seed{s}.unit{K}.jsonl"
        if f.exists():
            rows = f.read_text().splitlines()
            keep_rows = [r for r in rows if r.strip() and json.loads(r)["decision"] != d]
            if len(keep_rows) != len([r for r in rows if r.strip()]):
                shutil.copy(f, arch / f"{f.name}.{host}.{time.strftime('%H%M%S')}")
                f.write_text("".join(r + "\n" for r in keep_rows))
        for z in (od / "rec").glob(f"seed{s}_d{d}_c*_u{K}.npz"):
            z.rename(arch / f"{z.name}.{host}.moved")
        log(f"archived this runner's output of moved job {s}:{d}:K{K}")

    stopped = False
    while True:
        for pid, pgid, s, d, K in drivers(rp):
            if (s, d) in moved:
                os.killpg(pgid, signal.SIGTERM); log(f"killed driver of moved job {s}:{d}:K{K} (pgid {pgid})")
                time.sleep(2); archive_job(s, d, K)
        if not stopped and all((od / f"logs_{host}" / f"seed{s}_d{d}_u{K}.log").exists() for s, d, K in kept_jobs):
            if alive(rp):
                os.kill(rp, signal.SIGTERM)
            stopped = True; log("all kept jobs dispatched; runner stopped (its running drivers continue)")
        if stopped and not [x for x in drivers(rp) if (x[2], x[3]) not in moved]:
            break
        time.sleep(3)
    time.sleep(5)
    for pid, pgid, s, d, K in drivers(rp):
        os.killpg(pgid, signal.SIGKILL); archive_job(s, d, K)
    for sd in sdirs:
        pid = int(Path(sd, "pid").read_text().strip())
        if alive(pid):
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except OSError:
                os.kill(pid, signal.SIGTERM)
    log("servers stopped")

    def n_done(s, d, K):
        f = od / f"seed{s}.unit{K}.jsonl"
        if not f.exists():
            return 0
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        return len({r["cand"] for r in rows if r["decision"] == d and r.get("start_ok") and "success" in r})
    inc = [jb for jb in kept_jobs if n_done(*jb) != 8]
    summary = dict(host=host, jobs=len(kept_jobs), incomplete=sorted(inc), moved_to_bw=sorted(moved))
    (od / f"UNIT_DONE_{host}.json").write_text(json.dumps(summary) + "\n")
    log(f"done: {summary}")


if __name__ == "__main__":
    main()
