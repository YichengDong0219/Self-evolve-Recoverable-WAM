"""Does the candidate chunk itself carry the outcome, or is the sibling outcome split chance? (Claude, 2026-11-22
~16:10, post hoc: written after unit_result.txt; descriptive only.)
In the unit measurement every candidate i of a state has three continuations that share ONLY its first chunk:
  K=1: chunks 2.. use the shared seeds;  K=2: chunk 2 uses its own seed, then shared;  K=3: chunks 2-3 own seeds.
(K=2 and K=3 also share chunk 2, i.e. the first 96 steps.)
If the chosen chunk matters, the outcomes of the same candidate under different K agree more than expected from the
state's success rate. Statistics (states where both outcome vectors are mixed carry all the information):
  - co-success count vs a within-state permutation null (20000 permutations), one-sided p;
  - transfer of the oracle pick: candidates that succeeded under K=a, their success rate under K=b minus the state's
    mean under K=b (if the chunk carried the outcome this would approach 1 - mean).
Also: K=1 mixed-state share and +48 sibling spread by decision index (phase).
usage: python unit_candidate_effect.py --out-dir r0/unit_measure
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unit_common import FEATS  # noqa: E402

FI = {n: j for j, n in enumerate(FEATS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    od = a.out_dir
    states = [(x["seed"], x["decision"]) for x in json.loads((od / "unit_states.json").read_text())["states"]]
    rows = {}
    for f in od.glob("seed*.unit*.jsonl"):
        s, K = map(int, re.match(r"seed(\d+)\.unit(\d+)\.jsonl", f.name).groups())
        for r in map(json.loads, f.read_text().splitlines()):
            if r.get("start_ok") and "success" in r:
                rows[(s, r["decision"], r["cand"], K)] = float(r["success"])
    O = {K: np.array([[rows[(s, d, i, K)] for i in range(8)] for s, d in states]) for K in (1, 2, 3)}
    rng = np.random.default_rng(0)
    lines = ["candidate effect (post hoc, descriptive)"]
    for ka, kb in ((1, 2), (1, 3), (2, 3)):
        A, B = O[ka], O[kb]
        m = (A.sum(1) > 0) & (A.sum(1) < 8) & (B.sum(1) > 0) & (B.sum(1) < 8)
        A_, B_ = A[m], B[m]
        stat = float((A_ * B_).sum())
        null = np.array([(A_ * np.array([rng.permutation(b) for b in B_])).sum() for _ in range(20000)])
        p = (np.sum(null >= stat) + 1) / (len(null) + 1)
        r = [np.corrcoef(x, y)[0, 1] for x, y in zip(A_, B_)]
        # transfer of the oracle pick, both directions, on states mixed under the picking K
        tr = []
        for X, Y in ((A, B), (B, A)):
            mx = (X.sum(1) > 0) & (X.sum(1) < 8)
            for x, y in zip(X[mx], Y[mx]):
                tr.append((y[x == 1].mean() - y.mean(), 1 - y.mean()))
        tr = np.array(tr)
        shared = "first chunk only" if ka == 1 else "first two chunks"
        lines.append(f"K{ka} vs K{kb} (share {shared}): both-mixed states {int(m.sum())}; co-success {stat:.0f} vs null "
                     f"{null.mean():.1f} (p {p:.4f}); mean within-state r {np.mean(r):+.2f}; oracle pick transfer "
                     f"{tr[:, 0].mean():+.3f} of possible {tr[:, 1].mean():+.3f} (n {len(tr)})")
    # phase: decision index buckets
    for lo, hi in ((0, 2), (3, 5), (6, 9)):
        idx = [j for j, (s, d) in enumerate(states) if lo <= d <= hi]
        T = O[1][idx]; mixed = ((T.sum(1) > 0) & (T.sum(1) < 8)).sum()
        sd = []
        for j in idx:
            s, d = states[j]
            fp = []
            for i in range(8):
                c = np.load(od / "rec" / f"seed{s}_d{d}_c{i}_u1.npz")["caps"]
                c = c[c[:, FI["rel"]] <= 48][-1]
                fp.append(c[[FI["fp_x"], FI["fp_y"], FI["fp_z"]]])
            sd.append(np.sqrt((np.array(fp).std(0) ** 2).sum()) * 100)
        lines.append(f"decisions {lo}-{hi}: states {len(idx)}, K1 mixed {int(mixed)}, success {T.mean():.3f}, "
                     f"+48 mug spread median {np.median(sd):.2f} cm")
    out = "\n".join(lines) + "\n"
    (od / "unit_candidate_effect.txt").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
