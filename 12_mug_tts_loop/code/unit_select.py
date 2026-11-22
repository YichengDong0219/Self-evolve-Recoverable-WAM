"""States for the multi-chunk unit measurement (Claude, 2026-11-22; PROTOCOL addendum 3). Any python with numpy.
From the 296 gate states (gate_pairs.json + gate_pairs_retest.json), keep decisions <= 9 (at decision >= 10 no
candidate succeeded with K = 1), draw 54 without replacement with numpy default_rng(0), sort by (seed, decision).
usage: python unit_select.py --round-dir r0 --out r0/unit_measure/unit_states.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--round-dir", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--n", type=int, default=54); ap.add_argument("--max-decision", type=int, default=9)
a = ap.parse_args()
pool = []
for f in ("gate_pairs.json", "gate_pairs_retest.json"):
    pool += [(s["seed"], s["decision"]) for s in json.loads((a.round_dir / f).read_text())["states"]]
pool = sorted(set(pool))
elig = [s for s in pool if s[1] <= a.max_decision]
pick = sorted(elig[j] for j in np.random.default_rng(0).choice(len(elig), a.n, replace=False))
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(dict(note="PROTOCOL addendum 3: 54 of the gate states with decision <= 9, default_rng(0)",
                                 pool=len(pool), eligible=len(elig), states=[dict(seed=s, decision=d) for s, d in pick])))
print(len(pool), len(elig), pick)
