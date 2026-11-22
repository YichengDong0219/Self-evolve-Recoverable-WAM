"""Post-hoc decomposition of the multi-chunk unit measurement (Claude, 2026-11-22 15:30, written AFTER seeing
unit_result.txt; descriptive only, it does not change the pre-registered conclusion of PROTOCOL addendum 3).
Questions: (1) how much of the unit-end BoN gain only comes from candidates that already succeeded inside the unit
(both scores rank such a candidate first); (2) the scores' gain on the states where no candidate succeeded inside the
unit; (3) the per-state and per-scene gains of PRM phi on the mixed states (5 of the 11 K = 3 mixed states are one
scene, so a state bootstrap is optimistic; scene-level sign test added).
Same inputs and definitions as unit_analysis.py. usage: python unit_decompose.py --round-dir r0 --out-dir r0/unit_measure
"""
import argparse
import json
import re
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unit_common import FEATS  # noqa: E402

FI = {n: j for j, n in enumerate(FEATS)}


def goal(c):
    if c[FI["eval_success"]] > 0.5:
        return 0.0
    dx, dy = c[FI["fp_x"]] - c[FI["mid_x"]], c[FI["fp_y"]] - c[FI["mid_y"]]
    return float(np.sqrt(max(0, abs(dx) - 0.02) ** 2 + max(0, abs(dy) - 0.02) ** 2 + max(0, 0.86 - c[FI["fp_z"]]) ** 2))


def bon(t, s):
    top = np.isclose(s, np.max(s))
    return float(t[top].mean() - t.mean())


def boot(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    rng = np.random.default_rng(0)
    return np.percentile([rng.choice(x, len(x)).mean() for _ in range(n)], [2.5, 97.5]).round(3).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    od = a.out_dir
    states = [(x["seed"], x["decision"]) for x in json.loads((od / "unit_states.json").read_text())["states"]]
    prm = json.loads((od / "unit_prm.json").read_text())
    bw = {tuple(x) for x in json.loads((od / "unit_reassign.json").read_text())["to_bw"]}
    rows = {}
    for f in od.glob("seed*.unit*.jsonl"):
        s, K = map(int, re.match(r"seed(\d+)\.unit(\d+)\.jsonl", f.name).groups())
        for r in map(json.loads, f.read_text().splitlines()):
            if r.get("start_ok") and "success" in r:
                rows[(s, r["decision"], r["cand"], K)] = r
    lines = ["post-hoc decomposition (descriptive; the pre-registered rule is unchanged)"]
    for K in (1, 2, 3):
        end = 48 * K
        T = np.array([[float(rows[(s, d, i, K)]["success"]) for i in range(8)] for s, d in states])
        W = np.array([[1.0 if rows[(s, d, i, K)]["success_rel"] is not None and rows[(s, d, i, K)]["success_rel"] <= end
                       else 0.0 for i in range(8)] for s, d in states])
        P = np.array([[(prm.get(f"{s}_{d}_{i}_K{K}_r{end}") or [1.0])[0] for i in range(8)] for s, d in states], float)
        G = []
        for s, d in states:
            g = []
            for i in range(8):
                c = np.load(od / "rec" / f"seed{s}_d{d}_c{i}_u{K}.npz")["caps"]
                g.append(-goal(c[c[:, FI["rel"]] <= end][-1]))
            G.append(g)
        G = np.array(G)
        gW = np.array([bon(t, w) for t, w in zip(T, W)])
        gP = np.array([bon(t, p) for t, p in zip(T, P)])
        gG = np.array([bon(t, g) for t, g in zip(T, G)])
        mixed = (T.sum(1) > 0) & (T.sum(1) < 8); anyW = W.max(1) > 0; nw = ~anyW
        lines.append(f"K={K} (unit end +{end}): mixed {int(mixed.sum())} (with a within-unit success {int((mixed & anyW).sum())}), "
                     f"states with any within-unit success {int(anyW.sum())}")
        lines.append(f"   gain over {len(states)} states: within-unit success only {gW.mean():+.3f} {boot(gW)} | "
                     f"goal {gG.mean():+.3f} {boot(gG)} | prm {gP.mean():+.3f} {boot(gP)}")
        lines.append(f"   states without a within-unit success (n={int(nw.sum())}, mixed {int((mixed & nw).sum())}): "
                     f"goal {gG[nw].mean():+.3f} {boot(gG[nw])} | prm {gP[nw].mean():+.3f} {boot(gP[nw])}")
        ms = [states[j] for j in np.where(mixed)[0]]
        lines.append("   mixed states (seed, decision, gpu): prm gain / goal gain: " + ", ".join(
            f"({s},{d},{'pro6000' if (s, d) in bw else '4090'}) {gP[j]:+.3f}/{gG[j]:+.3f}"
            for (s, d), j in zip(ms, np.where(mixed)[0])))
        sc = {}
        for (s, d), j in zip(ms, np.where(mixed)[0]):
            sc.setdefault(s, []).append(gP[j])
        sm = {s: float(np.mean(v)) for s, v in sc.items()}
        pos, neg = sum(v > 0 for v in sm.values()), sum(v < 0 for v in sm.values())
        n = pos + neg
        p_one = sum(comb(n, k) for k in range(pos, n + 1)) / 2 ** n if n else None
        lines.append(f"   per-scene mean prm gain on mixed states: {{{', '.join(f'{s}: {v:+.3f}' for s, v in sm.items())}}}; "
                     f"scenes + {pos} / - {neg}, one-sided sign test p = {p_one:.4f}" if n else "   no scenes")
    out = "\n".join(lines) + "\n"
    (od / "unit_decompose.txt").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
