"""Hand-off plan for the PRO 6000 runner (Claude, 2026-11-22 ~14:00; user: "能否在Pro6000上也并行"). System python3, on Amax.
Fixed by queue position before any outcome is looked at: for each 4090 runner (Amax j % 9 in 0..4, cluster 5..8), the
states none of whose 3 jobs has started form the tail of its FIFO queue; tail states are moved to the PRO 6000 runner
one at a time (from the runner with the later projected finish) while that lowers the projected overall finish.
Projection (minutes): Amax 5.6 min/job on 5 slots, cluster 7.0 min/job on 4 slots, PRO 6000 7.5 min/job on 8 slots
+ 6 min start-up; running jobs count as half a job. Writes <out-dir>/unit_reassign.json.
"""
import json
import subprocess
import time
from pathlib import Path

OD = Path("/workspace/fact_local/mug_loop_1121/r0/unit_measure")
COD = "/workspace_cluster/fact_local/mug_loop_1121/r0/unit_measure"
st = [(x["seed"], x["decision"]) for x in json.loads((OD / "unit_states.json").read_text())["states"]]
la = set(p.name for p in (OD / "logs_Amax-Server").iterdir())
lc = set(subprocess.run(f"ssh -n -o BatchMode=yes $USER@CLUSTER_HOST 'ls {COD}/logs_user'", shell=True,
                        capture_output=True, text=True, timeout=60).stdout.split())
done_a = sum(1 for l in (OD / "unit_run_Amax-Server.log").read_text().splitlines() if " job " in l)
done_c = int(subprocess.run(f"ssh -n -o BatchMode=yes $USER@CLUSTER_HOST 'grep -c \" job \" {COD}/unit_run_user.log'",
                            shell=True, capture_output=True, text=True, timeout=60).stdout.strip() or 0)
side = {"amax": [s for j, s in enumerate(st) if j % 9 in {0, 1, 2, 3, 4}], "cluster": [s for j, s in enumerate(st) if j % 9 in {5, 6, 7, 8}]}
logs = {"amax": la, "cluster": lc}
started = lambda sd, s: any(f"seed{s[0]}_d{s[1]}_u{K}.log" in logs[sd] for K in (1, 2, 3))  # noqa: E731
running = {"amax": len(la) - done_a, "cluster": len(lc) - done_c}
free = {sd: [s for s in side[sd] if not started(sd, s)] for sd in side}
pending = {sd: 3 * len(free[sd]) + sum(3 - sum(f"seed{s[0]}_d{s[1]}_u{K}.log" in logs[sd] for K in (1, 2, 3))
                                       for s in side[sd] if started(sd, s)) for sd in side}
rate = {"amax": 5.6 / 5, "cluster": 7.0 / 4}
moved = {"amax": [], "cluster": []}


def t(sd):
    return (pending[sd] - 3 * len(moved[sd]) + 0.5 * running[sd]) * rate[sd]


def tb(extra=0):
    n = sum(len(v) for v in moved.values()) + extra
    return 6 + 3 * n * 7.5 / 8 if n else 0


while True:
    sd = max(side, key=t)
    if len(moved[sd]) >= len(free[sd]):
        break
    if tb(1) >= max(t("amax"), t("cluster")):
        break
    moved[sd].insert(0, free[sd][len(free[sd]) - 1 - len(moved[sd])])
plan = dict(note="13:58 hand-off to the PRO 6000 runner (user: 能否在Pro6000上也并行); fixed by queue position before any "
                 "outcome is looked at; all 3 units of a state stay on one runner",
            written=time.strftime("%F %T"), running=running, pending_jobs=pending,
            projected_min=dict(amax=round(t("amax")), cluster=round(t("cluster")), pro6000=round(tb())),
            amax_keep=[s for s in side["amax"] if s not in moved["amax"]], amax_moved=moved["amax"],
            cluster_keep=[s for s in side["cluster"] if s not in moved["cluster"]], cluster_moved=moved["cluster"],
            to_bw=moved["amax"] + moved["cluster"])
(OD / "unit_reassign.json").write_text(json.dumps(plan))
print(json.dumps({k: v for k, v in plan.items() if k not in ("amax_keep", "cluster_keep")}))
