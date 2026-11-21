"""V and Q (critic) training + gate evaluation (Claude, 2026-11-21; protocol 12, steps 3-5). CPU only (numpy).
Inputs per episode in --round-dir: <ep>.json (TTS episode), <ep>.exec.json (offline-executed candidates),
<ep>.prm.json (PRM features), optional <ep>.cont.jsonl (gate continuations: true outcome per candidate).
  V: logistic regression on [phi, hop_prev, t_frac, grip_left, grip_right] of trajectory states d,
     soft target G_d = gamma^(T-d) if the episode succeeds in chunk T = (steps-1)//48, else 0.
  Q: linear on standardized [hop_cur_to_im48, phi_im48, phi_im24, dq_left, dq_right, dgrip_left, dgrip_right];
     target y = 1 if the candidate succeeds within its chunk else gamma * V(s'), s' = its real +48 state with
     features [phi_re48, hop_cur_to_re48, 48(d+1)/step_lim, end grippers]; loss = listwise softmax CE (tau 0.1) + 0.1 MSE.
     L2 strength by 5-fold CV grouped by scene (within-state pair accuracy on y, |gap| >= 0.1).
  Gate (--mode gate): train on non-held-out scenes, evaluate BoN-8 on the held-out gate states with true outcomes.
  Fit (--mode fit): train on all scenes with the CV-chosen L2, write the critic JSON for prm_service.py.
usage: python critic_train.py --round-dir <dir> --mode gate|fit --out <json> [--gamma 0.9]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prm_common import Q_FEATURES, action_features  # noqa: E402

PLAN, TAU = 48, 0.1
L2S = [1e-3, 1e-2, 1e-1, 1.0]


def load_round(rd, gamma):
    eps = []
    for epf in sorted(rd.glob("hanging_mug_seed*.json")):
        if epf.name.endswith((".exec.json", ".prm.json")):
            continue
        pf, xf = epf.with_suffix(".prm.json"), epf.with_suffix(".exec.json")
        if not (pf.exists() and xf.exists()):
            continue
        ep = json.loads(epf.read_text()); prm = json.loads(pf.read_text())["features"]; ex = json.loads(xf.read_text())
        z = np.load(epf.with_suffix(".tts.npz"))
        cont = {}
        cf = epf.with_suffix(".cont.jsonl")
        if cf.exists():
            for r in map(json.loads, cf.read_text().splitlines()):
                if r.get("start_ok"):
                    cont[(r["decision"], r["cand"])] = float(r["success"])
        eps.append(dict(seed=ep["env_seed"], ep=ep, prm=prm, ex=ex, z=z, cont=cont))
    return eps


def grips(cap):
    g = cap.get("gripper_val", [0.0, 0.0])
    return [float(g[0]), float(g[1])]


def v_rows(e, gamma):
    ep, prm = e["ep"], e["prm"]
    D = e["z"]["cands"].shape[0]
    T = (ep["steps"] - 1) // PLAN
    X, y = [], []
    for d in range(D):
        cap = ep["states"].get(str(d * PLAN))
        if cap is None:
            continue
        X.append([prm["phi_cur"][str(d)], prm.get("hop_prev", {}).get(str(d), 0.0), d * PLAN / ep["step_lim"]] + grips(cap))
        y.append(gamma ** (T - d) if (ep["success"] and d <= T) else 0.0)
    return X, y


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def fit_logreg(X, y, l2, iters=3000, lr=0.1):
    X = np.asarray(X, float); y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(iters):
        p = sig(Z @ w + b); g = p - y
        w -= lr * (Z.T @ g / len(y) + l2 * w); b -= lr * g.mean()
    return dict(mean=mu.tolist(), std=sd.tolist(), w=w.tolist(), b=float(b))


def pred_logreg(m, X):
    Z = (np.asarray(X, float) - np.asarray(m["mean"])) / np.asarray(m["std"])
    return sig(Z @ np.asarray(m["w"]) + m["b"])


def auc(score, label):
    s, l = np.asarray(score), np.asarray(label).astype(bool)
    pos, neg = s[l], s[~l]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def q_groups(e, V, gamma):
    """One group per decision state: (features [N, F], targets [N], true outcomes or None, state id)."""
    ep, prm, ex, z = e["ep"], e["prm"], e["ex"], e["z"]
    rows = {(r["decision"], r["cand"]): r for r in ex["rows"]}
    D, N = z["cands"].shape[:2]
    out = []
    for d in range(D):
        F, Y, TR = [], [], []
        okd = True
        for i in range(N):
            r = rows.get((d, i)); key = f"{d}_{i}"
            if r is None or not r.get("start_ok") or key not in prm.get("phi_re48", {}):
                okd = False; break
            f = [prm["hop_cur_to_im48"][key], prm["phi_im48"][key], prm["phi_im24"][key]] + action_features(z["state"][d], z["cands"][d, i])
            vx = [prm["phi_re48"][key], prm["hop_cur_to_re48"][key], (d + 1) * PLAN / ep["step_lim"]] + grips(r["end_capture"])
            Y.append(1.0 if r["success_within_chunk"] else gamma * float(pred_logreg(V, [vx])[0]))
            F.append(f); TR.append(e["cont"].get((d, i)))
        if okd:
            out.append(dict(seed=e["seed"], d=d, F=np.nan_to_num(np.asarray(F, float)), Y=np.asarray(Y), true=TR))
    return out


def fit_q(groups, l2, iters=3000, lr=0.05):
    F = np.concatenate([g["F"] for g in groups]); mu, sd = F.mean(0), F.std(0) + 1e-6
    w, b = np.zeros(F.shape[1]), 0.0
    for _ in range(iters):
        gw, gb = np.zeros_like(w), 0.0
        for g in groups:
            Z = (g["F"] - mu) / sd; q = Z @ w + b
            p = np.exp((q - q.max())); p /= p.sum()
            t = np.exp((g["Y"] - g["Y"].max()) / TAU); t /= t.sum()
            gq = (p - t) + 0.1 * 2 * (q - g["Y"]) / len(q)
            gw += Z.T @ gq; gb += gq.sum()
        w -= lr * (gw / len(groups) + l2 * w); b -= lr * gb / len(groups)
    return dict(features=Q_FEATURES, mean=mu.tolist(), std=sd.tolist(), w=w.tolist(), b=float(b), l2=l2)


def q_pred(m, F):
    return ((np.asarray(F) - np.asarray(m["mean"])) / np.asarray(m["std"])) @ np.asarray(m["w"]) + m["b"]


def pair_acc(groups, m, gap=0.1):
    ok = []
    for g in groups:
        s = q_pred(m, g["F"])
        for a in range(len(s)):
            for b in range(len(s)):
                if g["Y"][a] - g["Y"][b] >= gap:
                    ok.append(s[a] > s[b])
    return float(np.mean(ok)) if ok else None


def cv_l2(groups):
    seeds = sorted({g["seed"] for g in groups}); rng = np.random.default_rng(0); rng.shuffle(seeds)
    fold = {s: i % 5 for i, s in enumerate(seeds)}
    best, best_acc = None, -1
    for l2 in L2S:
        accs = []
        for f in range(5):
            tr = [g for g in groups if fold[g["seed"]] != f]; te = [g for g in groups if fold[g["seed"]] == f]
            a = pair_acc(te, fit_q(tr, l2))
            if a is not None:
                accs.append(a)
        if accs and np.mean(accs) > best_acc:
            best, best_acc = l2, float(np.mean(accs))
    return best, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True); ap.add_argument("--mode", choices=["gate", "fit"], required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--gamma", type=float, default=0.9)
    a = ap.parse_args()
    eps = load_round(a.round_dir, a.gamma)
    seeds = sorted(e["seed"] for e in eps)
    held = set(seeds[::5]) if a.mode == "gate" else set()
    tr_eps = [e for e in eps if e["seed"] not in held]
    VX, Vy, Vsucc = [], [], []
    for e in tr_eps:
        X, y = v_rows(e, a.gamma); VX += X; Vy += y; Vsucc += [float(e["ep"]["success"])] * len(y)
    V = fit_logreg(VX, Vy, 1e-2)
    report = dict(mode=a.mode, episodes=len(eps), train_episodes=len(tr_eps), held_out=sorted(held), gamma=a.gamma,
                  V=V, V_auc_train=auc(pred_logreg(V, VX), Vsucc))
    if held:
        HX, Hs = [], []
        for e in eps:
            if e["seed"] in held:
                X, y = v_rows(e, a.gamma); HX += X; Hs += [float(e["ep"]["success"])] * len(y)
        report["V_auc_heldout"] = auc(pred_logreg(V, HX), Hs) if HX else None
    groups = [g for e in tr_eps for g in q_groups(e, V, a.gamma)]
    l2, cv = cv_l2(groups)
    Q = fit_q(groups, l2)
    report.update(Q=Q, Q_l2=l2, Q_cv_pair_acc=cv, Q_train_states=len(groups))
    if a.mode == "gate":
        rng = np.random.default_rng(0)
        hg = [g for e in eps if e["seed"] in held for g in q_groups(e, V, a.gamma)]
        gate = [g for g in hg if all(t is not None for t in g["true"])]
        gains, rnd, pk, orc = [], [], [], []
        for g in gate:
            tvals = np.asarray(g["true"], float); s = q_pred(Q, g["F"])
            k = int(np.argmax(s)); gains.append(tvals[k] - tvals.mean()); pk.append(tvals[k]); rnd.append(tvals.mean()); orc.append(tvals.max())
        boot = [np.mean(rng.choice(gains, len(gains))) for _ in range(2000)] if gains else []
        report.update(gate_states=len(gate), gate_picked=float(np.mean(pk)) if pk else None,
                      gate_random=float(np.mean(rnd)) if rnd else None, gate_oracle=float(np.mean(orc)) if orc else None,
                      gate_gain=float(np.mean(gains)) if gains else None,
                      gate_gain_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None)
        report["gate_pass"] = bool(report["gate_gain_ci"] and report["gate_gain_ci"][0] > 0)
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("V", "Q")}, indent=1))


if __name__ == "__main__":
    main()
