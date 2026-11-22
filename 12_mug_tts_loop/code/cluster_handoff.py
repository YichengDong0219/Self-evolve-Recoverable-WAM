"""Hand the tail of the cluster's re-test queue to Amax (Claude, 2026-11-22 12:40; user: "让Amax分担集群的工作").
Runs ON THE CLUSTER, system python3. Reads r0/gate_retest_reassign.json {"cluster_keep": [[s,d],...],
"amax_extra": [[s,d],...], "gate_run_pid": P, "server_pids": [...]}.
Assignment is fixed by queue position before any outcome is looked at: the cluster keeps the first 7 not-yet-started
states of its FIFO queue, Amax runs the last 10. Every state's rows come from its assigned machine only.
 1. poll every 3 s; once every cluster_keep state has been dispatched (its driver log exists), SIGTERM the cluster
    gate_run.py (its running drivers are separate processes and keep running to completion);
 2. any driver of an amax_extra state on the cluster (dispatch race) is killed by its process group;
 3. when no driver of this gate_run session is left, stop the cluster's probe servers by PID, archive cluster
    logs/rows of amax_extra states, and write GATE_CONT_DONE_user.json (states, incomplete).
Never touches other users' processes (only PIDs whose session id is the gate_run's, and the listed servers).
"""
import json
import os
import shutil
import signal
import time
from pathlib import Path

RD = Path("/workspace_cluster/fact_local/mug_loop_1121/r0")
LOGD = RD / "gate_logs_user"
ARCH = RD / "_gate_retest_archive"
LOG = RD / "cluster_handoff.log"


def log(m):
    with LOG.open("a") as f:
        f.write(time.strftime("%F %T ") + m + "\n")


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def drivers(sid):
    """(pid, pgid, seed, decision) of continue_candidates.py processes in session sid owned by me."""
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            args = (p / "cmdline").read_bytes().split(b"\0")
            stat = (p / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        pgid, s = int(stat[2]), int(stat[3])
        a = [x.decode() for x in args if x]
        if s != sid or not any(x.endswith("continue_candidates.py") for x in a) or "timeout" in a[0]:
            continue
        ep = a[a.index("--episode") + 1]; d = int(a[a.index("--pairs") + 1].split(":")[0])
        out.append((int(p.name), pgid, int(Path(ep).name[16:22]), d))
    return out


def rows(seed):
    f = RD / f"hanging_mug_seed{seed}.cont.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def main():
    cfg = json.loads((RD / "gate_retest_reassign.json").read_text())
    keep = [tuple(x) for x in cfg["cluster_keep"]]; extra = {tuple(x) for x in cfg["amax_extra"]}
    gp = cfg["gate_run_pid"]; ARCH.mkdir(exist_ok=True)
    log(f"start: keep {len(keep)} extra {len(extra)} gate_run {gp}")
    killed_parent = False
    while True:
        for pid, pgid, s, d in drivers(gp):
            if (s, d) in extra:
                os.killpg(pgid, signal.SIGTERM); log(f"killed cluster driver of reassigned state {s}:{d} (pgid {pgid})")
        if not killed_parent and all((LOGD / f"seed{s}_d{d}.log").exists() for s, d in keep):
            if alive(gp):
                os.kill(gp, signal.SIGTERM)
            killed_parent = True; log("all kept states dispatched; cluster gate_run stopped")
        if killed_parent and not [x for x in drivers(gp) if (x[2], x[3]) not in extra]:
            break
        time.sleep(3)
    time.sleep(5)
    for pid, pgid, s, d in drivers(gp):          # a reassigned driver that survived the TERM
        os.killpg(pgid, signal.SIGKILL); log(f"SIGKILL {s}:{d}")
    for pid in cfg["server_pids"]:
        if alive(pid):
            os.kill(pid, signal.SIGTERM)
    log("servers stopped")
    for s, d in sorted(extra):
        lg = LOGD / f"seed{s}_d{d}.log"
        if lg.exists():
            lg.rename(ARCH / f"{lg.name}.reassigned_to_amax")
        rs = rows(s); kept_rows = [r for r in rs if r["decision"] != d]
        if len(kept_rows) != len(rs):
            f = RD / f"hanging_mug_seed{s}.cont.jsonl"
            shutil.copy(f, ARCH / f"{f.name}.handoff{time.strftime('%H%M%S')}")
            f.write_text("".join(json.dumps(r) + "\n" for r in kept_rows))
            log(f"dropped {len(rs) - len(kept_rows)} cluster rows of reassigned state {s}:{d}")
    st = [(x["seed"], x["decision"]) for x in json.loads((RD / "gate_pairs_retest.json").read_text())["states"]]
    mine = [x for j, x in enumerate(st) if j % 9 in {5, 6, 7, 8} and x not in extra]
    inc = [x for x in mine if len({r["cand"] for r in rows(x[0]) if r["decision"] == x[1]}) != 8]
    summary = dict(host="user", states=len(mine), incomplete=sorted(inc), reassigned_to_amax=sorted(extra))
    (RD / "GATE_CONT_DONE_user.json").write_text(json.dumps(summary))
    log(f"done: {summary}")


if __name__ == "__main__":
    main()
