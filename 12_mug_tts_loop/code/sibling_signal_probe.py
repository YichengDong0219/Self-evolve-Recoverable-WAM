"""Exploratory, descriptive only (Claude, 2026-11-22 ~13:40): which information can rank the 8 sibling candidates of a
decision state by their TRUE continuation outcome? Uses the 296 gate states (20 held-out scenes; after this they are
development data, future gates need fresh scenes). CPU only, numpy.
Labels: true outcome of one pi0 continuation per candidate (<ep>.cont.jsonl).
Model per feature set: the same linear listwise scorer as critic_train.fit_q, but trained on the TRUE outcomes,
leave-one-scene-out (train 19 scenes, score the 20th), fixed L2 0.01 (chosen a priori, not tuned).
Feature sets:
  Q7        the current Q inputs (imagined PRM x3 + action stats x4), i.e. what is available at decision time
  ACT4      action stats only;  IMAG3  imagined PRM only
  PRMREAL2  PRM on the REAL +48 frames (phi, hop)
  PRIV8     privileged sim state after REALLY executing the candidate 48 steps (mug height, mug displacement and
            rotation during the chunk, mug speed, mug-rack xy distance and height difference, gripper openings)
  PRIV8+PRMREAL2
Metrics: pair accuracy on within-state pairs with different true outcomes (mixed states), BoN-8 gain over all states;
bootstrap CI over states (default_rng(0), 2000); permutation null = labels shuffled within each state, refit (40x).
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, "/workspace/fact_local/mug_loop_1121/code")
import critic_train as C  # noqa: E402
from round_select import held_out, kept_seeds  # noqa: E402

RD = Path("/workspace/fact_local/mug_loop_1121/r0")
PLAN, L2, NPERM = 48, 0.01, 40


def actor(cap, name):
    for k, v in cap["actors"].items():
        if k.endswith(name):
            return v
    raise KeyError(name)


def priv(start_cap, end_cap):
    m0, m1, rk = actor(start_cap, "039_mug"), actor(end_cap, "039_mug"), actor(end_cap, "040_rack")
    p0, p1, r = np.array(m0["p"]), np.array(m1["p"]), np.array(rk["p"])
    q0, q1 = np.array(m0["q"]), np.array(m1["q"])
    ang = 2 * np.arccos(min(1.0, abs(float(q0 @ q1))))
    g = end_cap.get("gripper_val", [0.0, 0.0])
    return [p1[2], np.linalg.norm(p1 - p0), ang, np.linalg.norm(m1.get("v", [0, 0, 0])),
            np.linalg.norm(p1[:2] - r[:2]), p1[2] - r[2], float(g[0]), float(g[1])]


PRIV_NAMES = ["mug_z", "mug_disp", "mug_rot", "mug_speed", "mug_rack_xy", "mug_minus_rack_z", "grip_l", "grip_r"]


def build():
    crit = json.loads((RD / "critic_gate.json").read_text())
    kept, _ = kept_seeds(RD, 170000); held = held_out(kept)
    eps, _ = C.load_round(RD, held)
    want = set()
    for f in ("gate_pairs.json", "gate_pairs_retest.json"):
        want |= {(s["seed"], s["decision"]) for s in json.loads((RD / f).read_text())["states"]}
    out = []
    for e in eps:
        rows = {(r["decision"], r["cand"]): r for r in e["ex"]["rows"]}
        for g in C.q_groups(e, crit["V"], crit["gamma"]):
            if (g["seed"], g["d"]) not in want:
                continue
            sc = e["ep"]["states"][str(g["d"] * PLAN)]
            P = np.array([priv(sc, rows[(g["d"], i)]["end_capture"]) for i in range(8)], float)
            out.append(dict(seed=g["seed"], d=g["d"], true=np.array(g["true"], float), Q7=g["F"], ACT4=g["F"][:, 3:],
                            IMAG3=g["F"][:, :3], PRMREAL2=g["RE"], PRIV8=P, **{"PRIV8+PRMREAL2": np.hstack([P, g["RE"]])}))
    return sorted(out, key=lambda x: (x["seed"], x["d"]))


def loso(states, key, labels):
    scenes = sorted({s["seed"] for s in states}); score = [None] * len(states)
    for sc in scenes:
        tr = [dict(F=s[key], Y=labels[j]) for j, s in enumerate(states) if s["seed"] != sc]
        m = C.fit_q(tr, L2)
        for j, s in enumerate(states):
            if s["seed"] == sc:
                score[j] = C.q_pred(m, s[key])
    return score


def metrics(states, labels, score):
    pa, gains = [], []
    for t, s in zip(labels, score):
        k = int(np.argmax(s)); gains.append(t[k] - t.mean())
        ok = [1.0 if s[a] > s[b] else 0.5 if s[a] == s[b] else 0.0 for a in range(8) for b in range(8) if t[a] > t[b]]
        pa.append(np.mean(ok) if ok else None)
    mixed = [x for x in pa if x is not None]
    return float(np.mean(mixed)), float(np.mean(gains)), mixed, gains


def run(key):
    states = build()
    labels = [s["true"] for s in states]
    sc = loso(states, key, labels)
    acc, gain, mixed, gains = metrics(states, labels, sc)
    rng = np.random.default_rng(0)
    ci_acc = np.percentile([np.mean(rng.choice(mixed, len(mixed))) for _ in range(2000)], [2.5, 97.5]).tolist()
    ci_gain = np.percentile([np.mean(rng.choice(gains, len(gains))) for _ in range(2000)], [2.5, 97.5]).tolist()
    pr = np.random.default_rng(1); null = []
    for _ in range(NPERM):
        lab = [pr.permutation(t) for t in labels]
        null.append(metrics(states, lab, loso(states, key, lab))[0])
    return dict(set=key, pair_acc=acc, pair_acc_ci=ci_acc, bon_gain=gain, bon_gain_ci=ci_gain,
                null_pair_acc_mean=float(np.mean(null)), null_pair_acc_95=float(np.percentile(null, 95)),
                perm_p=float((np.sum(np.array(null) >= acc) + 1) / (NPERM + 1)), mixed_states=len(mixed))


def describe():
    st = build(); mixed = [s for s in st if 0 < s["true"].sum() < 8]
    print(f"states {len(st)}, mixed {len(mixed)}, scenes {len({s['seed'] for s in st})}")
    P = np.stack([s["PRIV8"] for s in st])
    print("privileged state after +48, sibling spread vs across-state spread:")
    for j, n in enumerate(PRIV_NAMES):
        print(f"  {n:17s} within-state SD {P[:, :, j].std(1).mean():.4f}   across-state SD {P[:, :, j].mean(1).std():.4f}")
    print("mixed states, mean (success - failure) candidate difference:")
    for j, n in enumerate(PRIV_NAMES):
        dif = [s["PRIV8"][s["true"] == 1, j].mean() - s["PRIV8"][s["true"] == 0, j].mean() for s in mixed]
        print(f"  {n:17s} {np.mean(dif):+.4f} (positive in {np.mean(np.array(dif) > 0):.0%} of mixed states)")


if __name__ == "__main__":
    describe()
    keys = ["Q7", "ACT4", "IMAG3", "PRMREAL2", "PRIV8", "PRIV8+PRMREAL2"]
    with Pool(len(keys)) as p:
        res = p.map(run, keys)
    for r in res:
        print(json.dumps(r))
