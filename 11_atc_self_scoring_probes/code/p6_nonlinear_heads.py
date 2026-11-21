"""ATC P6 (Claude, 2026-11-21; user approved "先跑第1步的CPU探针"): are the frozen WAM features exhausted by a linear head?
Protocol: 11_atc_self_scoring_probes/PROTOCOL.md section P6. CPU only.
Features (mean over the K = 4 draws): R2 and released-FACT fs, and all = fs+val+img; standardised, PCA-128 fitted on
train states. Heads: 2-layer MLP (128 hidden, ReLU, dropout 0.2), n-weighted MSE, Adam lr 1e-3, 300 full-batch epochs,
weight decay in {1e-3, 1e-2, 1e-1} by 5-fold state-grouped CV (within-state pair accuracy, gap >= 0.5), 5 seeds averaged;
plus a state-centred variant (features and labels minus their state mean). Linear reference: weighted ridge on the same
raw features (lambda by the same CV). Test metrics on the 53 test states: pair_acc, BoN-8 (+ bootstrap CI), reliable
pairs (both n >= 4), and the paired BoN difference head - ridge.
usage: python p6_nonlinear_heads.py <wm_value/v2 dir> <FACT feats npz> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

V2, F_FACT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
torch.set_num_threads(16)
rows = [json.loads(l) for l in (V2 / "chunk_rows.jsonl").read_text().splitlines() if l.strip()]
ids = [r["id"] for r in rows]
y = np.asarray([r["label"] for r in rows]); w = np.asarray([r["n"] for r in rows], float)
st = np.asarray([r["state"] for r in rows]); split = np.asarray([r["split"] for r in rows])
strat = np.asarray([r["stratum"] for r in rows]); kind = np.asarray([r["kind"] for r in rows])
TR, TE = np.where(split == "train")[0], np.where(split == "test")[0]
WDS = [1e-3, 1e-2, 1e-1]; LAMBDAS = [10.0 ** e for e in range(-1, 6)]; SEEDS = range(5)


def load(p):
    z = np.load(p); assert [str(s) for s in z["side"]] == ids
    f = {k: z[f"feat_{k}"].astype(np.float32).mean(1) for k in ("fs", "val", "img")}
    return {"fs": f["fs"], "all": np.concatenate([f["fs"], f["val"], f["img"]], 1)}


FEATS = {"R2": load(V2 / "feats_chunks.npz"), "FACT": load(F_FACT)}


def by_state(idx):
    out = {}
    for i in idx:
        out.setdefault(st[i], []).append(i)
    return {k: np.asarray(v) for k, v in out.items()}


def pairs_of(idx, gap=0.5, min_n=0):
    return [(a, b) for ii in by_state(idx).values() for a in ii for b in ii
            if y[a] - y[b] >= gap and w[a] >= min_n and w[b] >= min_n]


def pair_acc(s, idx, min_n=0):
    P = pairs_of(idx, min_n=min_n)
    return (round(float(np.mean([s[a] > s[b] for a, b in P])), 3) if P else None), len(P)


def bon_per_state(s, idx):
    g = by_state([i for i in idx if kind[i] == "policy"])
    keys = sorted(g)
    gain = []
    for k in keys:
        ii = g[k]; sc = s[ii]; top = ii[sc == sc.max()]
        gain.append(y[top].mean() - y[ii].mean())
    return keys, np.asarray(gain)


def boot(v, B=2000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def pca_fit(Xtr, k=128):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = (Xtr - mu) / sd
    _, _, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    comp = vt[:k].T
    return lambda X: ((X - mu) / sd) @ comp


def centre(X, t, idx):
    Xc, tc = X.copy(), t.copy()
    for k, ii in by_state(idx).items():
        Xc[ii] -= X[ii].mean(0); tc[ii] -= np.average(t[ii], weights=w[ii])
    return Xc, tc


def mlp_fit_predict(Xtr, ttr, wtr, Xte, wd, seed):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 128), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    X = torch.tensor(Xtr, dtype=torch.float32); T = torch.tensor(ttr, dtype=torch.float32); W = torch.tensor(wtr, dtype=torch.float32)
    for _ in range(300):
        opt.zero_grad(); loss = (W * (net(X).squeeze(1) - T) ** 2).sum() / W.sum(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Xte, dtype=torch.float32)).squeeze(1).numpy()


def mlp_scores(X, tr, te, wd, centred):
    proj = pca_fit(X[tr]); P = np.zeros((len(rows), 128), np.float32); P[tr] = proj(X[tr]); P[te] = proj(X[te])
    t = y.copy()
    if centred:
        Xtr, ttr = centre(P, t, tr); Xte, _ = centre(P, t, te)   # test labels unused; only features are centred
        Xtr, ttr, Xte = Xtr[tr], ttr[tr], Xte[te]
    else:
        Xtr, ttr, Xte = P[tr], t[tr], P[te]
    return np.mean([mlp_fit_predict(Xtr, ttr, w[tr], Xte, wd, s) for s in SEEDS], 0)


def wridge_scores(X, tr, te, lam):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xa, Xb = (X[tr] - mu) / sd, (X[te] - mu) / sd
    sw = np.sqrt(w[tr])[:, None]; Xw = Xa * sw; tw = (y[tr] - np.average(y[tr], weights=w[tr])) * sw[:, 0]
    alpha = np.linalg.solve(Xw @ Xw.T + lam * np.eye(len(Xa)), tw)
    return Xb @ (Xw.T @ alpha)


def folds():
    sts = sorted(set(st[TR])); rng = np.random.default_rng(0); rng.shuffle(sts)
    f = {s: i % 5 for i, s in enumerate(sts)}
    return [np.asarray([i for i in TR if f[st[i]] == j]) for j in range(5)]


def cv_select(fit, grid):
    best, best_acc = None, -1
    for g in grid:
        ok = []
        for fo in folds():
            sc = np.full(len(rows), np.nan); sc[fo] = fit(np.setdiff1d(TR, fo), fo, g)
            ok += [sc[a] > sc[b] for a, b in pairs_of(fo)]
        acc = float(np.mean(ok))
        if acc > best_acc:
            best, best_acc = g, acc
    return best, round(best_acc, 3)


def evaluate(s):
    keys, gain = bon_per_state(s, TE)
    out = dict(pair_acc_test=pair_acc(s, TE), reliable_pairs_test=pair_acc(s, TE, min_n=4),
               bon8_gain=round(float(gain.mean()), 3), bon8_ci=boot(gain))
    for lab in ("place", "recovery"):
        m = np.asarray([strat[by_state([i for i in TE if kind[i] == "policy"])[k][0]] == lab for k in keys])
        out[f"bon8_gain_{lab}"] = round(float(gain[m].mean()), 3)
    return out, keys, gain


def main():
    res = {}
    for model, F in FEATS.items():
        for fname, X in F.items():
            lam, cv_r = cv_select(lambda tr, te, g: wridge_scores(X, tr, te, g), LAMBDAS)
            s_r = np.full(len(rows), np.nan); s_r[TE] = wridge_scores(X, TR, TE, lam)
            ev_r, keys, g_r = evaluate(s_r)
            res[f"{model}/{fname}/ridge"] = dict(lambda_=lam, cv=cv_r, **ev_r)
            for centred in (False, True):
                tag = "mlp_c" if centred else "mlp"
                wd, cv_m = cv_select(lambda tr, te, g: mlp_scores(X, tr, te, g, centred), WDS)
                s_m = np.full(len(rows), np.nan); s_m[TE] = mlp_scores(X, TR, TE, wd, centred)
                ev_m, keys_m, g_m = evaluate(s_m)
                assert keys_m == keys
                d = g_m - g_r
                res[f"{model}/{fname}/{tag}"] = dict(wd=wd, cv=cv_m, **ev_m, bon8_minus_ridge=round(float(d.mean()), 3),
                                                     bon8_minus_ridge_ci=boot(d))
                print(model, fname, tag, json.dumps(res[f"{model}/{fname}/{tag}"]), flush=True)
            print(model, fname, "ridge", json.dumps(res[f"{model}/{fname}/ridge"]), flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
