"""ATC P5 (Claude, 2026-11-21): hanging_mug seed census over all existing seeds100 evaluations. CPU only.
Is a mug failure decided by the scene (no policy ever solves the seed -> test-time selection cannot help) or by sampling?
Inputs: <eval dir>/<arm>_nv_step<k>/hanging_mug_seed<s>.json (fact-g0-episode/1, field "success"); partial checkpoints skipped.
seeds100 = the first 100 valid seeds from 160000 (invalid ones are replaced by 1601xx), so the seed set is the union of
the seeds actually evaluated (11-21 fix: the first version assumed 160000-160099 and dropped the replacement seeds).
usage: python p5_mug_seed_census.py <out.json> <eval dir> [<eval dir> ...]
"""
import glob
import json
import os
import re
import sys

import numpy as np

OUT, DIRS = sys.argv[1], sys.argv[2:]


def main():
    ck = {}
    for base in DIRS:
        for a in sorted(os.listdir(base)):
            m = re.fullmatch(r"mug(\w+?)_nv_step(\d+)", a)
            if not m:
                continue
            res = {}
            for f in glob.glob(os.path.join(base, a, "hanging_mug_seed*.json")):
                d = json.load(open(f))
                res[int(d["env_seed"])] = bool(d["success"])
            ck[a] = dict(arm=m.group(1), step=int(m.group(2)), res=res)
    names = sorted(ck)
    SEEDS = sorted(set().union(*[set(ck[n]["res"]) for n in names]))
    M = np.full((len(SEEDS), len(names)), np.nan)
    for j, n in enumerate(names):
        for i, s in enumerate(SEEDS):
            if s in ck[n]["res"]:
                M[i, j] = ck[n]["res"][s]

    def census(cols, label):
        X = M[:, cols]
        cnt = np.nansum(X, 1); tot = np.sum(~np.isnan(X), 1); rate = cnt / np.maximum(tot, 1)
        valid = ~np.isnan(X)
        allv = X[valid]; seed_mean = np.repeat(rate[:, None], X.shape[1], 1)[valid]
        return dict(label=label, checkpoints=len(cols), seeds=len(SEEDS), evaluated_per_seed_min=int(tot.min()), mean_success=round(float(np.nanmean(X)), 3),
                    seeds_never_solved=int(np.sum(cnt == 0)), seeds_always_solved=int(np.sum(cnt == tot)),
                    seeds_solved_by_at_least_one=int(np.sum(cnt > 0)),
                    rate_hist={f"{lo:.1f}-{lo + 0.2:.1f}": int(np.sum((rate >= lo) & (rate < lo + 0.2 + (1e-9 if lo >= 0.8 else 0))))
                               for lo in (0.0, 0.2, 0.4, 0.6, 0.8)},
                    between_seed_var_frac=round(float(seed_mean.var() / allv.var()), 3) if allv.var() > 0 else None,
                    per_checkpoint_success={names[j]: int(np.nansum(M[:, j])) for j in cols})

    all_cols = list(range(len(names)))
    final = {}
    for j, n in enumerate(names):
        a = ck[n]["arm"]
        if a not in final or ck[n]["step"] > ck[names[final[a]]]["step"]:
            final[a] = j
    out = dict(all_checkpoints=census(all_cols, "all checkpoints"),
               final_checkpoints=census(sorted(final.values()), "final checkpoint of each arm"),
               checkpoints=names)
    open(OUT, "w").write(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_checkpoint_success"} if isinstance(v, dict) else v
                      for k, v in out.items() if k != "checkpoints"}, indent=1))


if __name__ == "__main__":
    main()
