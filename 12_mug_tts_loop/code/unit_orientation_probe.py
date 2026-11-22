"""Mug orientation vs position: can either separate the 8 siblings of the K = 1 mixed unit states (leave-one-out
nearest-centroid within each state, states with >= 2 successes and >= 2 failures)? (Claude, 2026-11-22 ~16:45, post hoc,
descriptive; the sibling probe PRIV8 had no absolute mug orientation.) Run in mug_loop_1121: python code/unit_orientation_probe.py"""
import json, re, sys, numpy as np
from pathlib import Path
sys.path.insert(0, "code")
from unit_common import FEATS
FI = {n: j for j, n in enumerate(FEATS)}
od = Path("r0/unit_measure")
states = [(x["seed"], x["decision"]) for x in json.loads((od / "unit_states.json").read_text())["states"]]
rows = {}
for f in od.glob("seed*.unit1.jsonl"):
    s = int(f.name[4:10])
    for r in map(json.loads, f.read_text().splitlines()):
        if r.get("start_ok") and "success" in r: rows[(s, r["decision"], r["cand"])] = float(r["success"])
def cap(s, d, i, rel):
    c = np.load(od / "rec" / f"seed{s}_d{d}_c{i}_u1.npz")["caps"]
    return c[c[:, FI["rel"]] <= rel][-1]
def qang(a, b):
    return 2 * np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b))))))
def qmean(Q):
    Q = np.array(Q); ref = Q[0]; Q = np.array([q if np.dot(q, ref) >= 0 else -q for q in Q]); m = Q.mean(0); return m / np.linalg.norm(m)
mixed = [(s, d) for s, d in states if 0 < sum(rows[(s, d, i)] for i in range(8)) < 8]
print("K=1 mixed states", len(mixed))
for rel in (48, 96, 144, 192, 240, 288):
    accO, accP, spO = [], [], []
    for s, d in mixed:
        T = np.array([rows[(s, d, i)] for i in range(8)])
        C = [cap(s, d, i, rel) for i in range(8)]
        Q = [c[[FI["mug_qw"], FI["mug_qx"], FI["mug_qy"], FI["mug_qz"]]] / np.linalg.norm(c[[FI["mug_qw"], FI["mug_qx"], FI["mug_qy"], FI["mug_qz"]]]) for c in C]
        P = [c[[FI["fp_x"], FI["fp_y"], FI["fp_z"]]] for c in C]
        m = qmean(Q); spO.append(np.median([qang(q, m) for q in Q]))
        if T.sum() < 2 or (8 - T.sum()) < 2: continue
        sO, sP = np.zeros(8), np.zeros(8)
        for i in range(8):
            S = [j for j in range(8) if j != i and T[j] == 1]; F = [j for j in range(8) if j != i and T[j] == 0]
            sO[i] = qang(Q[i], qmean([Q[j] for j in F])) - qang(Q[i], qmean([Q[j] for j in S]))
            sP[i] = np.linalg.norm(P[i] - np.mean([P[j] for j in F], 0)) - np.linalg.norm(P[i] - np.mean([P[j] for j in S], 0))
        for sc, acc in ((sO, accO), (sP, accP)):
            pr = [1.0 if sc[a] > sc[b] else 0.5 if sc[a] == sc[b] else 0.0 for a in range(8) for b in range(8) if T[a] > T[b]]
            acc.append(np.mean(pr))
    print(f"rel {rel:3d}: mug orientation spread median {np.median(spO):5.2f} deg | LOO nearest-centroid pair acc "
          f"(states with >=2 succ and >=2 fail, n={len(accO)}): orientation {np.mean(accO):.2f}  position {np.mean(accP):.2f}")
