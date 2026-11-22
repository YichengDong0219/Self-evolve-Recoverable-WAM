"""Per decision point of the round-0 trajectory: how many of the 8 candidates succeed (gate continuations, K = 1),
and how many succeed inside their own chunk (Claude, 2026-11-22 ~16:40, post hoc, descriptive; user: "决定能否成功的
就是挂杯子那一下"). Run in mug_loop_1121 with any python (json only): python code/gate_phase_table.py r0"""
import collections
import glob
import json
import sys
from pathlib import Path
rd = Path(sys.argv[1]); rows = collections.defaultdict(dict)
for f in sorted(glob.glob(str(rd / "hanging_mug_seed*.cont.jsonl"))):
    s = int(Path(f).name[16:22])
    for r in map(json.loads, open(f).read().splitlines()):
        if r.get("start_ok") and "success" in r:
            rows[s][(r["decision"], r["cand"])] = (bool(r["success"]), r["steps"])
print("seed round0 steps | successes of 8 at decision 0..8 (wN = N of them succeed inside their own chunk)")
for s in sorted(rows):
    ep = json.load(open(rd / f"hanging_mug_seed{s}.json")); cells = []
    for d in range(9):
        if all((d, c) in rows[s] for c in range(8)):
            n = sum(rows[s][(d, c)][0] for c in range(8))
            w = sum(rows[s][(d, c)][0] and rows[s][(d, c)][1] <= 48 * (d + 1) for c in range(8))
            cells.append(f"{n}" + (f"w{w}" if w else ""))
        else:
            cells.append("-")
    print(s, "OK  " if ep["success"] else "FAIL", ep["steps"], " ".join(f"{c:>4}" for c in cells))
