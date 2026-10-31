"""Takeover step for a failed hanging_mug rollout from its stored 48-step boundary states (Claude, 2026-10-28).

  dropped  the mug reached hook height (z0 + 10 cm) and later came back below z0 + 5 cm: the first boundary after
           the last hook-height boundary where it is low again (the fall has happened; recovery waits for rest)
  stalled  it never reached hook height: the first boundary >= 192 from which the mug pose stays within 5 mm of its
           final pose until the end (the policy gave up)
  None     anything else (e.g. still high at the end, or never settles): not used as a recovery state
"""
import json
import sys

import numpy as np


def mug_z_series(src):
    st = src["states"]
    keys = sorted((int(k) for k in st if k != "final"))
    pos = [np.asarray(next(v for n, v in st[str(k)]["actors"].items() if "039_mug" in n)["p"], float) for k in keys]
    fin = np.asarray(next(v for n, v in st["final"]["actors"].items() if "039_mug" in n)["p"], float)
    return keys, pos, fin


def takeover_step(src):
    keys, pos, fin = mug_z_series(src)
    z0 = pos[0][2]
    high = [i for i, p in enumerate(pos) if p[2] > z0 + 0.10]
    if high:
        after = [i for i in range(high[-1] + 1, len(keys)) if pos[i][2] < z0 + 0.05]
        if after:
            return keys[after[0]], "dropped"
        return None, "still_high"
    for i, k in enumerate(keys):
        if k >= 192 and all(np.linalg.norm(pos[j] - fin) < 0.005 for j in range(i, len(keys))):
            return k, "stalled"
    return None, "never_settles"


if __name__ == "__main__":
    for p in sys.argv[1:]:
        d = json.load(open(p))
        print(p.split("/")[-1], d["success"], *takeover_step(d))
