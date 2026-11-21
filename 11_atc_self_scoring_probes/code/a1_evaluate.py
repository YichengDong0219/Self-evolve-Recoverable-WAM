"""ATC A1 pilot evaluation (Claude, 2026-11-21). Protocol: PROTOCOL.md section A1. CPU only.
Scores (lower value = better, score = -mean over draws; P6's best head added per the P6 rule): ATC-trained R2 checkpoints (G1b 20-step value), R2 before training
(same G1b path), released FACT 20-step value (G1b), and the linear head ridge/fs on frozen R2 features (lambda 1000,
fit on train states). Test metrics on the 53 test states: pair_acc (gap >= 0.5), reliable pairs (both n >= 4), BoN-8 gain
with state bootstrap CI, and the paired difference "ATC - ridge/fs" (the A1 criterion). Train-state metrics for ATC
show the fit (overfitting check).
usage: python a1_evaluate.py <wm_value/v2 dir> <FACT d20 npz> <R2-before g1b npz> <FACT feats npz> <out.json> <name=ATC g1b npz> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np

V2, F_D20, F_R2B, F_FACT_FEATS, OUT = map(Path, sys.argv[1:6])
ATC = [a.split("=", 1) for a in sys.argv[6:]]
rows = [json.loads(l) for l in (V2 / "chunk_rows.jsonl").read_text().splitlines() if l.strip()]
ids = [r["id"] for r in rows]
y = np.asarray([r["label"] for r in rows]); w = np.asarray([r["n"] for r in rows], float)
st = np.asarray([r["state"] for r in rows]); split = np.asarray([r["split"] for r in rows])
strat = np.asarray([r["stratum"] for r in rows]); kind = np.asarray([r["kind"] for r in rows])
TR, TE = np.where(split == "train")[0], np.where(split == "test")[0]


def val_score(p, key):
    z = np.load(p); assert [str(s) for s in z["side"]] == ids, p
    return -z[key].astype(np.float64).mean(1)


def by_state(idx):
    out = {}
    for i in idx:
        out.setdefault(st[i], []).append(i)
    return {k: np.asarray(v) for k, v in out.items()}


def pair_acc(s, idx, min_n=0):
    P = [(a, b) for ii in by_state(idx).values() for a in ii for b in ii if y[a] - y[b] >= 0.5 and w[a] >= min_n and w[b] >= min_n]
    return [round(float(np.mean([s[a] > s[b] for a, b in P])), 3) if P else None, len(P)]


def bon(s, idx):
    g = by_state([i for i in idx if kind[i] == "policy"]); keys = sorted(g); gain = []
    for k in keys:
        ii = g[k]; sc = s[ii]; top = ii[sc == sc.max()]; gain.append(y[top].mean() - y[ii].mean())
    return keys, np.asarray(gain)


def boot(v, B=2000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def ridge_fs():
    z = np.load(V2 / "feats_chunks.npz"); X = z["feat_fs"].astype(np.float32).mean(1)
    mu, sd = X[TR].mean(0), X[TR].std(0) + 1e-6; Xa, Xb = (X[TR] - mu) / sd, (X[TE] - mu) / sd
    sw = np.sqrt(w[TR])[:, None]; Xw = Xa * sw; tw = (y[TR] - np.average(y[TR], weights=w[TR])) * sw[:, 0]
    alpha = np.linalg.solve(Xw @ Xw.T + 1000.0 * np.eye(len(Xa)), tw)
    s = np.full(len(rows), np.nan); s[TE] = Xb @ (Xw.T @ alpha); return s


def p6_head(feats_npz, wd=0.1, seeds=range(5)):
    """P6's best head by the protocol rule: released-FACT 'all' features, state-centred MLP (wd 0.1 selected by CV in P6);
    same code as p6_nonlinear_heads.py (PCA-128 on train, 2x128 MLP, dropout 0.2, Adam 1e-3, 300 epochs, 5 seeds)."""
    import torch
    torch.set_num_threads(16)
    z = np.load(feats_npz); assert [str(x) for x in z["side"]] == ids
    X = np.concatenate([z[f"feat_{k}"].astype(np.float32).mean(1) for k in ("fs", "val", "img")], 1)
    mu, sd = X[TR].mean(0), X[TR].std(0) + 1e-6; Z = (X[TR] - mu) / sd
    _, _, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False); comp = vt[:128].T
    P = ((X - mu) / sd) @ comp
    def centre(idx, t):
        Xc, tc = P.copy(), t.copy()
        for k, ii in by_state(idx).items():
            Xc[ii] -= P[ii].mean(0); tc[ii] -= np.average(t[ii], weights=w[ii])
        return Xc, tc
    Xtr, ttr = centre(TR, y.copy()); Xte, _ = centre(TE, y.copy())
    preds = []
    for sd_ in seeds:
        torch.manual_seed(sd_)
        net = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
        Xt = torch.tensor(Xtr[TR], dtype=torch.float32); Tt = torch.tensor(ttr[TR], dtype=torch.float32); Wt = torch.tensor(w[TR], dtype=torch.float32)
        for _ in range(300):
            opt.zero_grad(); loss = (Wt * (net(Xt).squeeze(1) - Tt) ** 2).sum() / Wt.sum(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            preds.append(net(torch.tensor(Xte[TE], dtype=torch.float32)).squeeze(1).numpy())
    s = np.full(len(rows), np.nan); s[TE] = np.mean(preds, 0); return s


def report(s, ref_gain=None, ref_keys=None, with_train=False):
    keys, gain = bon(s, TE)
    out = dict(pair_acc_test=pair_acc(s, TE), reliable_pairs_test=pair_acc(s, TE, 4), bon8_gain=round(float(gain.mean()), 3),
               bon8_ci=boot(gain))
    pol = by_state([i for i in TE if kind[i] == "policy"])
    for lab in ("place", "recovery"):
        m = np.asarray([strat[pol[k][0]] == lab for k in keys]); out[f"bon8_gain_{lab}"] = round(float(gain[m].mean()), 3)
    if ref_gain is not None:
        assert keys == ref_keys
        d = gain - ref_gain; out["minus_ridge"] = round(float(d.mean()), 3); out["minus_ridge_ci"] = boot(d)
    if with_train:
        kt, gt = bon(s, TR); out["train_pair_acc"] = pair_acc(s, TR); out["train_bon8_gain"] = round(float(gt.mean()), 3)
    return out, keys, gain


def main():
    res = {}
    r_out, rk, rg = report(ridge_fs()); res["ridge/fs (R2 frozen features)"] = r_out
    res["P6 best head (FACT all, centred MLP)"] = report(p6_head(F_FACT_FEATS), rg, rk)[0]
    res["R2 before (20-step value)"] = report(val_score(F_R2B, "value_final"), rg, rk, True)[0]
    res["released FACT (20-step value)"] = report(val_score(F_D20, "value_pred"), rg, rk, True)[0]
    for name, p in ATC:
        res[f"ATC {name} (20-step value)"] = report(val_score(p, "value_final"), rg, rk, True)[0]
        res[f"ATC {name} (step-0 value)"] = report(val_score(p, "value_step0"), rg, rk, True)[0]
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for k, v in res.items():
        print(f"{k:40s} {json.dumps(v)}")


if __name__ == "__main__":
    main()
