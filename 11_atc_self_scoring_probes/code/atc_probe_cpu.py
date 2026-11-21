"""ATC CPU probes P1-P4 (Claude, 2026-11-21; user: "先做一些不用GPU/轻量GPU的探针实验"). CPU only, no fitting of the WAM.

Definitions are frozen in 11_atc_self_scoring_probes/PROTOCOL.md (same metrics as the 11-14 wm_value_head2.py).
  P1  value decomposition: does the WAM value rank states (between) or candidates within a state (within)?
  P2  perfect-consequence ceiling: score candidates by the TRUE simulator state at the end of the chunk
      (a) fixed rule 10*[success now] - box excess - 0.1*distance, (b) weighted ridge on those features
  P3  headroom: state-level pass@1 vs pass@8 of the 8 policy branches; chunk-level random vs oracle
  P4  label-noise-free selection: among n = 5 candidates, select on 2 outcomes, evaluate on the other 3
usage: python atc_probe_cpu.py <wm_value/v2 dir> <chunk_end.jsonl> <out.json> [<feats npz> <tag>]
       (the optional feats/tag pair re-runs the value-based parts on another model's features, e.g. original FACT)
"""
import json
import sys
from pathlib import Path

import numpy as np

V2, END, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
FEATS = Path(sys.argv[4]) if len(sys.argv) > 4 else V2 / "feats_chunks.npz"
TAG = sys.argv[5] if len(sys.argv) > 5 else "R2"
LAMBDAS = [10.0 ** e for e in range(-1, 6)]
EPS1, EPS2 = 0.015, 0.012          # place_mouse_pad check_success box (x, y)
N_BOOT, N_SPLIT = 2000, 1000

rows = [json.loads(l) for l in (V2 / "chunk_rows.jsonl").read_text().splitlines() if l.strip()]
ids = [r["id"] for r in rows]
z = np.load(FEATS)
assert [str(s) for s in z["side"]] == ids, "feature order != chunk rows"
end = {}
for l in END.read_text().splitlines():
    if l.strip():
        e = json.loads(l); end[e["id"]] = e
assert set(end) == set(ids)
y = np.asarray([r["label"] for r in rows], np.float64)
wgt = np.asarray([r["n"] for r in rows], np.float64)
st = np.asarray([r["state"] for r in rows])
split = np.asarray([r["split"] for r in rows])
strat = np.asarray([r["stratum"] for r in rows])
kind = np.asarray([r["kind"] for r in rows])
own = np.asarray([bool(r["own"]) if r["kind"] == "policy" else False for r in rows])
TR, TE = np.where(split == "train")[0], np.where(split == "test")[0]
ALL = np.arange(len(rows))
POL = np.where(kind == "policy")[0]
vp = z["value_pred"].astype(np.float64).mean(1)          # FACT value: lower is better
S_VALUE = -vp
F_FS = z["feat_fs"].astype(np.float32).mean(1)


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def rankdata(a):
    a = np.asarray(a, np.float64); order = np.argsort(a, kind="mergesort"); ranks = np.empty(len(a)); sa = a[order]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def by_state(idx):
    out = {}
    for i in idx:
        out.setdefault(st[i], []).append(i)
    return {k: np.asarray(v) for k, v in out.items()}


def pairs_of(idx, gap=0.5):
    return [(a, b) for ii in by_state(idx).values() for a in ii for b in ii if y[a] - y[b] >= gap]


def pair_acc(score, idx):
    P = pairs_of(idx)
    res = {}
    for name, sel in (("all", None), ("place", "place"), ("recovery", "recovery")):
        Q = [(a, b) for a, b in P if sel is None or strat[a] == sel]
        ok = [score[a] > score[b] for a, b in Q]
        res[name] = dict(acc=r3(np.mean(ok)) if Q else None, pairs=len(Q), ties=int(sum(score[a] == score[b] for a, b in Q)))
    return res


def boot_ci(vals, rng):
    vals = np.asarray(vals, np.float64)
    if len(vals) == 0:
        return [None, None]
    m = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(N_BOOT)]
    return [r3(np.percentile(m, 2.5)), r3(np.percentile(m, 97.5))]


def best_of_8(score, idx):
    """Per state, among its policy chunks: label of the top-scored one (ties -> mean over the tied ones = expected
    value of a random tie-break), vs mean (random pick) vs max (oracle, noisy labels)."""
    rng = np.random.default_rng(0)
    res = {}
    groups = by_state([i for i in idx if kind[i] == "policy"])
    for name, sel in (("all", None), ("place", "place"), ("recovery", "recovery")):
        pk, rd, orc = [], [], []
        for s, ii in groups.items():
            if sel and strat[ii[0]] != sel:
                continue
            sc = score[ii]; top = ii[sc == sc.max()]
            pk.append(y[top].mean()); rd.append(y[ii].mean()); orc.append(y[ii].max())
        if not pk:
            continue
        gain = np.asarray(pk) - np.asarray(rd); og = np.asarray(orc) - np.asarray(rd)
        res[name] = dict(states=len(pk), picked=r3(np.mean(pk)), random=r3(np.mean(rd)), oracle=r3(np.mean(orc)),
                         gain=r3(gain.mean()), gain_ci=boot_ci(gain, rng), oracle_gain=r3(og.mean()),
                         frac_of_oracle=r3(gain.mean() / og.mean()) if og.mean() > 0 else None)
    return res


def standardise(Xa, Xb):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    return (Xa - mu) / sd, (Xb - mu) / sd


def wridge(X, t, w, lam):
    sw = np.sqrt(w)[:, None]; Xw = X * sw; tw = (t - np.average(t, weights=w)) * sw[:, 0]
    if X.shape[1] < X.shape[0]:
        return np.linalg.solve(Xw.T @ Xw + lam * np.eye(X.shape[1]), Xw.T @ tw)
    alpha = np.linalg.solve(Xw @ Xw.T + lam * np.eye(len(X)), tw)
    return Xw.T @ alpha


def ridge_fit_predict(X, tr, te, lam):
    Xa, Xb = standardise(X[tr], X[te])
    return Xb @ wridge(Xa, y[tr], wgt[tr], lam)


def folds_by_state(idx, k=5):
    sts = sorted(set(st[idx])); rng = np.random.default_rng(0); rng.shuffle(sts)
    f = {s: i % k for i, s in enumerate(sts)}
    return [np.asarray([i for i in idx if f[st[i]] == j]) for j in range(k)]


def cv_lambda(X, idx):
    best, best_acc = None, -1
    for lam in LAMBDAS:
        ok = []
        for fo in folds_by_state(idx):
            trn = np.setdiff1d(idx, fo)
            sc = np.full(len(rows), np.nan); sc[fo] = ridge_fit_predict(X, trn, fo, lam)
            ok += [sc[a] > sc[b] for a, b in pairs_of(fo)]
        acc = float(np.mean(ok))
        if acc > best_acc:
            best, best_acc = lam, acc
    return best, round(best_acc, 3)


def oof_and_test(X, lam):
    """Out-of-fold predictions on train states + predictions on test states from a fit on all train states."""
    sc = np.full(len(rows), np.nan)
    for fo in folds_by_state(TR):
        sc[fo] = ridge_fit_predict(X, np.setdiff1d(TR, fo), fo, lam)
    sc[TE] = ridge_fit_predict(X, TR, TE, lam)
    return sc


def decomposition(score, idx):
    """P1: between-state vs within-state agreement of a score with the labels (policy chunks only)."""
    rng = np.random.default_rng(0)
    g = by_state([i for i in idx if kind[i] == "policy"])
    keys = sorted(g)
    ms = np.asarray([score[g[k]].mean() for k in keys]); my = np.asarray([y[g[k]].mean() for k in keys])
    bet = spearman(ms, my)
    bs = []
    for _ in range(N_BOOT):
        b = rng.integers(0, len(keys), len(keys))
        v = spearman(ms[b], my[b])
        if not np.isnan(v):
            bs.append(v)
    within = [spearman(score[g[k]], y[g[k]]) for k in keys if len(set(y[g[k]])) > 1 and len(set(score[g[k]])) > 1]
    within = [v for v in within if not np.isnan(v)]

    def var_frac(v):
        v = np.asarray(v, np.float64); allv = np.concatenate([v[g[k]] for k in keys])
        bm = np.concatenate([np.full(len(g[k]), v[g[k]].mean()) for k in keys])
        return r3(bm.var() / allv.var()) if allv.var() > 0 else None

    return dict(states=len(keys), between_spearman=r3(bet),
                between_ci=[r3(np.percentile(bs, 2.5)), r3(np.percentile(bs, 97.5))],
                within_states=len(within), within_mean_spearman=r3(np.mean(within)) if within else None,
                within_ci=boot_ci(within, rng), within_frac_positive=r3(np.mean(np.asarray(within) > 0)) if within else None,
                between_var_frac_score=var_frac(score), between_var_frac_label=var_frac(y))


def end_features():
    cols = ["dx", "dy", "adx", "ady", "r", "ex", "held_left", "held_right", "released", "success_end", "dz", "orient"]
    X = np.zeros((len(rows), len(cols)), np.float64)
    for i, cid in enumerate(ids):
        e = end[cid]; dx, dy = e["dx"], e["dy"]
        ex = max(abs(dx) - EPS1, 0.0) + max(abs(dy) - EPS2, 0.0)
        X[i] = [dx, dy, abs(dx), abs(dy), np.hypot(dx, dy), ex, e["holder"] == "left", e["holder"] == "right",
                e["holder"] == "none", e["success_end"], e["dz"], e["orientation_error"]]
    return X, cols


def p4_noise_free():
    """Among each state's n = 5 policy candidates (>= 2 of them): pick by 2 outcomes, evaluate on the other 3."""
    rng = np.random.default_rng(0)
    g = by_state([i for i in POL if wgt[i] == 5])
    res = {}
    per = {}
    for s, ii in g.items():
        if len(ii) < 2:
            continue
        ks = [int(round(y[i] * 5)) for i in ii]
        vec = [np.asarray([1] * k + [0] * (5 - k)) for k in ks]
        ev = []
        for _ in range(N_SPLIT):
            sel, eva = [], []
            for v in vec:
                p = rng.permutation(v); sel.append(p[:2].sum()); eva.append(p[2:].mean())
            sel = np.asarray(sel); eva = np.asarray(eva)
            ev.append(eva[sel == sel.max()].mean())
        per[s] = dict(selected_2=float(np.mean(ev)), random=float(y[ii].mean()), naive_oracle=float(y[ii].max()),
                      k=len(ii), stratum=strat[ii[0]])
    for name, sel in (("all", None), ("place", "place"), ("recovery", "recovery")):
        P = [v for v in per.values() if sel is None or v["stratum"] == sel]
        if not P:
            continue
        g2 = np.asarray([v["selected_2"] - v["random"] for v in P]); go = np.asarray([v["naive_oracle"] - v["random"] for v in P])
        res[name] = dict(states=len(P), candidates_mean=r3(np.mean([v["k"] for v in P])),
                         random=r3(np.mean([v["random"] for v in P])), selected_by_2_outcomes=r3(np.mean([v["selected_2"] for v in P])),
                         naive_oracle=r3(np.mean([v["naive_oracle"] for v in P])), gain_selected_2=r3(g2.mean()),
                         gain_ci=boot_ci(g2, rng), gain_naive_oracle=r3(go.mean()))
    return res


def p3_headroom():
    res = {}
    g = by_state(POL)
    for name, sel in (("all", None), ("place", "place"), ("recovery", "recovery")):
        ks = [k for k in g if sel is None or strat[g[k][0]] == sel]
        p1 = [own[g[k]].mean() for k in ks]; p8 = [own[g[k]].any() for k in ks]
        rd = [y[g[k]].mean() for k in ks]; orc = [y[g[k]].max() for k in ks]
        res[name] = dict(states=len(ks), pass_at_1=r3(np.mean(p1)), pass_at_8=r3(np.mean(p8)),
                         states_all_fail=int(sum(not x for x in p8)), states_all_success=int(sum(v == 1.0 for v in p1)),
                         chunk_random=r3(np.mean(rd)), chunk_oracle_noisy=r3(np.mean(orc)))
    return res


def main():
    out = dict(tag=TAG, feats=FEATS.name, chunks=len(rows), states=len(set(st)), test_states=len(set(st[TE])))
    # value-based parts (rerun per model)
    lam_fs, cv_fs = cv_lambda(F_FS, TR)
    s_fs = oof_and_test(F_FS, lam_fs)
    out["value"] = dict(pair_acc_test=pair_acc(S_VALUE, TE), best_of_8_test=best_of_8(S_VALUE, TE),
                        pair_acc_train=pair_acc(S_VALUE, TR), best_of_8_train=best_of_8(S_VALUE, TR),
                        P1_decomposition_all_states=decomposition(S_VALUE, ALL))
    out["ridge_fs"] = dict(lambda_=lam_fs, cv_pair_acc=cv_fs, pair_acc_test=pair_acc(s_fs, TE), best_of_8_test=best_of_8(s_fs, TE),
                           P1_decomposition_oof_train_plus_test=decomposition(s_fs, ALL))
    out["P1_labels_reference"] = dict(between_var_frac_label=decomposition(y, ALL)["between_var_frac_label"])
    if TAG == "R2":
        X, cols = end_features()
        e = X[:, cols.index("ex")]; r = X[:, cols.index("r")]; succ = X[:, cols.index("success_end")]
        s_rule = 10.0 * succ - e - 0.1 * r
        lam_end, cv_end = cv_lambda(X, TR)
        s_end = np.full(len(rows), np.nan); s_end[TE] = ridge_fit_predict(X, TR, TE, lam_end)
        out["P2_rule"] = dict(pair_acc_test=pair_acc(s_rule, TE), best_of_8_test=best_of_8(s_rule, TE),
                              pair_acc_all=pair_acc(s_rule, ALL), best_of_8_all=best_of_8(s_rule, ALL))
        out["P2_ridge_end_features"] = dict(features=cols, lambda_=lam_end, cv_pair_acc=cv_end,
                                            pair_acc_test=pair_acc(s_end, TE), best_of_8_test=best_of_8(s_end, TE))
        held = X[:, cols.index("held_left")] + X[:, cols.index("held_right")]
        out["P2_chunk_end_census"] = {
            name: dict(chunks=int(m.sum()), success_at_end=int((succ * m).sum()), held_at_end=int((held * m).sum()),
                       mean_label=r3(y[m.astype(bool)].mean()))
            for name, m in (("place", (strat == "place").astype(float)), ("recovery", (strat == "recovery").astype(float)))}
        out["P3_headroom"] = p3_headroom()
        out["P4_noise_free_selection"] = p4_noise_free()
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
