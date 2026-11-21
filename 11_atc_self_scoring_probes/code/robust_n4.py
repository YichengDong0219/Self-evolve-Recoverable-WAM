"""ATC supplementary robustness check (Claude, 2026-11-21; user asked whether the probes assume the DPO pairs are right).
POST HOC, not in the frozen protocol. CPU only.
Re-computes the main scorer comparisons using only candidates whose label rests on >= 4 real continuations
(rechecked policy chunks n = 5, cuRobo chunks n = 4), instead of all candidates (most policy chunks have n = 1).
  pairs_n4   within-state pairs, both n >= 4, label gap >= 0.5, all 188 states (fitted heads: out-of-fold on train + test)
  bon_n5     per state, among its n = 5 policy candidates (>= 2): label of the top-scored vs mean vs max
usage: python robust_n4.py <wm_value/v2 dir> <chunk_end.jsonl> <R2 feats npz> <FACT feats npz> <FACT denoise20 npz> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

V2, END = Path(sys.argv[1]), Path(sys.argv[2])
F_R2, F_FACT, F_D20, OUT = map(Path, sys.argv[3:7])
EPS1, EPS2 = 0.015, 0.012
rows = [json.loads(l) for l in (V2 / "chunk_rows.jsonl").read_text().splitlines() if l.strip()]
ids = [r["id"] for r in rows]
y = np.asarray([r["label"] for r in rows]); n = np.asarray([r["n"] for r in rows], float)
st = np.asarray([r["state"] for r in rows]); split = np.asarray([r["split"] for r in rows])
strat = np.asarray([r["stratum"] for r in rows]); kind = np.asarray([r["kind"] for r in rows])
TR, TE = np.where(split == "train")[0], np.where(split == "test")[0]
end = {json.loads(l)["id"]: json.loads(l) for l in END.read_text().splitlines() if l.strip()}


def load(p):
    z = np.load(p); assert [str(s) for s in z["side"]] == ids
    return z


zr, zf, zd = load(F_R2), load(F_FACT), load(F_D20)


def standardise(Xa, Xb):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    return (Xa - mu) / sd, (Xb - mu) / sd


def wridge(X, t, w, lam):
    sw = np.sqrt(w)[:, None]; Xw = X * sw; tw = (t - np.average(t, weights=w)) * sw[:, 0]
    alpha = np.linalg.solve(Xw @ Xw.T + lam * np.eye(len(X)), tw)
    return Xw.T @ alpha


def oof_and_test(X, lam):
    sts = sorted(set(st[TR])); rng = np.random.default_rng(0); rng.shuffle(sts)
    f = {s: i % 5 for i, s in enumerate(sts)}
    sc = np.full(len(rows), np.nan)
    for j in range(5):
        fo = np.asarray([i for i in TR if f[st[i]] == j]); trn = np.setdiff1d(TR, fo)
        Xa, Xb = standardise(X[trn], X[fo]); sc[fo] = Xb @ wridge(Xa, y[trn], n[trn], lam)
    Xa, Xb = standardise(X[TR], X[TE]); sc[TE] = Xb @ wridge(Xa, y[TR], n[TR], lam)
    return sc


ex = np.asarray([max(abs(end[c]["dx"]) - EPS1, 0) + max(abs(end[c]["dy"]) - EPS2, 0) for c in ids])
rr = np.asarray([np.hypot(end[c]["dx"], end[c]["dy"]) for c in ids])
succ = np.asarray([float(end[c]["success_end"]) for c in ids])
SC = {
    "FACT value 20-step (G1b)": -zd["value_pred"].astype(float).mean(1),
    "FACT value one-step (G1)": -zf["value_pred"].astype(float).mean(1),
    "R2 value": -zr["value_pred"].astype(float).mean(1),
    "ridge on FACT fs features (OOF+test)": oof_and_test(zf["feat_fs"].astype(np.float32).mean(1), 100.0),
    "ridge on R2 fs features (OOF+test)": oof_and_test(zr["feat_fs"].astype(np.float32).mean(1), 1000.0),
    "P2 rule, true chunk-end position": 10 * succ - ex - 0.1 * rr,
}


def by_state(idx):
    out = {}
    for i in idx:
        out.setdefault(st[i], []).append(i)
    return out


def boot(v, rng, B=2000):
    v = np.asarray(v)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


res = {}
good = [i for i in range(len(rows)) if n[i] >= 4]
for name, s in SC.items():
    rng = np.random.default_rng(0)
    out = {}
    for lab, sel in (("all", None), ("place", "place"), ("recovery", "recovery")):
        P = [(a, b) for ii in by_state(good).values() for a in ii for b in ii if y[a] - y[b] >= 0.5 and (sel is None or strat[a] == sel)]
        ok = [s[a] > s[b] for a, b in P]
        g5 = {k: v for k, v in by_state([i for i in good if kind[i] == "policy"]).items() if len(v) >= 2}
        pk, rd, orc = [], [], []
        for k, ii in g5.items():
            if sel and strat[ii[0]] != sel:
                continue
            ii = np.asarray(ii); sc = s[ii]; top = ii[sc == sc.max()]
            pk.append(y[top].mean()); rd.append(y[ii].mean()); orc.append(y[ii].max())
        gain = np.asarray(pk) - np.asarray(rd)
        out[lab] = dict(pairs=len(P), pair_acc=round(float(np.mean(ok)), 3), states=len(pk), picked=round(float(np.mean(pk)), 3),
                        random=round(float(np.mean(rd)), 3), oracle=round(float(np.mean(orc)), 3),
                        gain=round(float(gain.mean()), 3), gain_ci=boot(gain, rng))
    res[name] = out
OUT.write_text(json.dumps(dict(note="post hoc robustness check, labels with >= 4 continuations only", results=res), indent=1))
for name, o in res.items():
    a = o["all"]
    print(f"{name:40s} pairs {a['pairs']:4d} acc {a['pair_acc']:.3f} | BoN states {a['states']} gain {a['gain']:+.3f} {a['gain_ci']} "
          f"(place {o['place']['gain']:+.3f}, recovery {o['recovery']['gain']:+.3f})")
