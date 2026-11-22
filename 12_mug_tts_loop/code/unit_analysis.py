"""Analysis of the multi-chunk unit measurement (Claude, 2026-11-22; PROTOCOL addendum 3, written before the run).
Any python with numpy. Inputs: <out-dir> (unit_states.json, seed*.unit<K>.jsonl, rec/*.npz, unit_prm.json,
excluded.json) and the round dir (gate continuations <ep>.cont.jsonl, exec end captures).
Per unit length K (a candidate unit = the candidate chunk + K-1 more pi0 chunks with candidate-specific seeds; pi0
with shared seeds afterwards; label = success at the end of the episode):
 a. selectable variance: share of mixed states, random / oracle success, oracle - random (bootstrap CI over states);
 b. physical divergence of the 8 siblings at rel = 48j: within-state SD (cm) of the mug functional point and of the
    right end-effector position, and of the right gripper value;
 c. separability on mixed states (pair accuracy on pairs with different outcomes, ties 0.5) at the unit end and at
    every rel: (i) unfitted goal distance (lower better): sqrt(max(0,|dx|-0.02)^2 + max(0,|dy|-0.02)^2 +
    max(0, 0.86 - z)^2), (dx, dy) = mug functional point - rack middle, z = functional point height, 0 once the task
    succeeded; (ii) PRM phi on the real frame (higher better; 1.0 once succeeded); (iii) LOSO linear listwise model on
    privileged features (descriptive);
 d. BoN-8 gain of (i) and (ii) at the unit end over all states, and per chunk (gain / K);
 e. K = 1 only: separability / gain of (i), (ii) at rel = 96..288 (score after 1 chunk is executed, look further).
Determinism: K = 1 rows vs the gate rows (action hash, outcome); captures of the shared prefix across K.
Recommendation rule (pre-registered): among K whose (i)-gain CI lower bound > 0, take the largest gain per chunk;
if its per-chunk CI overlaps that of a smaller qualifying K, take the smaller K; if no K qualifies, no unit length up
to 3 chunks makes the candidates distinguishable.
usage: python unit_analysis.py --round-dir r0 --out-dir r0/unit_measure
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import critic_train as C  # noqa: E402
from unit_common import FEATS  # noqa: E402

FI = {n: j for j, n in enumerate(FEATS)}
PLAN, UNITS, RELS = 48, (1, 2, 3), (48, 96, 144, 192, 240, 288)
PRIV = ["dx", "dy", "fp_z", "mug_z", "r_dx", "r_dy", "r_dz", "grip_r", "grip_l", "lee_z"]


def goal_dist(c):
    if c[FI["eval_success"]] > 0.5:
        return 0.0
    dx, dy = c[FI["fp_x"]] - c[FI["mid_x"]], c[FI["fp_y"]] - c[FI["mid_y"]]
    return float(np.sqrt(max(0, abs(dx) - 0.02) ** 2 + max(0, abs(dy) - 0.02) ** 2 + max(0, 0.86 - c[FI["fp_z"]]) ** 2))


def priv(c):
    return [c[FI["fp_x"]] - c[FI["mid_x"]], c[FI["fp_y"]] - c[FI["mid_y"]], c[FI["fp_z"]], c[FI["mug_z"]],
            c[FI["ree_x"]] - c[FI["fp_x"]], c[FI["ree_y"]] - c[FI["fp_y"]], c[FI["ree_z"]] - c[FI["fp_z"]],
            c[FI["grip_r"]], c[FI["grip_l"]], c[FI["lee_z"]]]


def boot(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(0)
    return np.percentile([np.mean(rng.choice(x, len(x))) for _ in range(n)], [2.5, 97.5]).tolist()


def pair_acc(T, S):
    """T, S: [n, 8]; per mixed state mean pair accuracy (ties 0.5) -> list."""
    out = []
    for t, s in zip(T, S):
        ok = [1.0 if s[a] > s[b] else 0.5 if s[a] == s[b] else 0.0 for a in range(8) for b in range(8) if t[a] > t[b]]
        if ok:
            out.append(float(np.mean(ok)))
    return out


def bon(T, S):
    g = []
    for t, s in zip(T, S):
        top = np.isclose(s, np.max(s))
        g.append(float(t[top].mean() - t.mean()))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    rd, od = a.round_dir, a.out_dir
    states = [(x["seed"], x["decision"]) for x in json.loads((od / "unit_states.json").read_text())["states"]]
    excl = {tuple(x) for x in json.loads((od / "excluded.json").read_text())} if (od / "excluded.json").exists() else set()
    prm = json.loads((od / "unit_prm.json").read_text()) if (od / "unit_prm.json").exists() else {}
    rows, caps = {}, {}
    for f in od.glob("seed*.unit*.jsonl"):
        s, K = map(int, re.match(r"seed(\d+)\.unit(\d+)\.jsonl", f.name).groups())
        for r in map(json.loads, f.read_text().splitlines()):
            if r.get("start_ok") and "success" in r and (s, r["decision"], K) not in excl:
                rows[(s, r["decision"], r["cand"], K)] = r
    for (s, d, i, K) in rows:
        caps[(s, d, i, K)] = np.load(od / "rec" / f"seed{s}_d{d}_c{i}_u{K}.npz")["caps"]

    def cap_at(key, rel):
        c = caps[key]; m = c[:, FI["rel"]] <= rel
        return c[m][-1]

    def prm_at(key, rel):
        s, d, i, K = key
        v = prm.get(f"{s}_{d}_{i}_K{K}_r{rel}")
        if v is None:
            r = rows[key]
            return (1.0, 1.0) if (r["success"] and r["success_rel"] is not None and r["success_rel"] <= rel) else (None, None)
        return tuple(v)

    res = dict(states=len(states), excluded=sorted(excl), per_K={}, determinism={})
    # determinism
    gate = {}
    for s in {s for s, _ in states}:
        f = rd / f"hanging_mug_seed{s}.cont.jsonl"
        for r in map(json.loads, f.read_text().splitlines()):
            if r.get("start_ok"):
                gate[(s, r["decision"], r["cand"])] = r
    bw = set()
    if (od / "unit_reassign.json").exists():       # 13:58 hand-off: these states ran with FACT on the PRO 6000s
        bw = {tuple(x) for x in json.loads((od / "unit_reassign.json").read_text())["to_bw"]}
    res["pro6000_states"] = sorted(bw)
    for tag, sel in (("4090", lambda k: k[:2] not in bw), ("pro6000", lambda k: k[:2] in bw)):
        kk = [(k, r) for k, r in rows.items() if k[3] == 1 and k[:3] in gate and sel(k)]
        res["determinism"][f"K1_vs_gate_{tag}"] = dict(
            pairs=len(kk), action_hash_equal=sum(r["cont_actions_sha"] == gate[k[:3]]["cont_actions_sha"] for k, r in kk),
            outcome_equal=sum(r["success"] == gate[k[:3]]["success"] for k, r in kk))
    k1 = [(k, r) for k, r in rows.items() if k[3] == 1 and k[:3] in gate]
    res["determinism"]["K1_vs_gate"] = dict(
        pairs=len(k1), action_hash_equal=sum(r["cont_actions_sha"] == gate[k[:3]]["cont_actions_sha"] for k, r in k1),
        outcome_equal=sum(r["success"] == gate[k[:3]]["success"] for k, r in k1),
        same_host_pairs=sum(r["host"] == gate[k[:3]]["host"] for k, r in k1),
        same_host_hash_equal=sum(r["host"] == gate[k[:3]]["host"] and r["cont_actions_sha"] == gate[k[:3]]["cont_actions_sha"]
                                 for k, r in k1))
    shared = []
    for (s, d, i, K) in rows:
        if K == 3 and (s, d, i, 2) in rows:
            a2, a3 = caps[(s, d, i, 2)], caps[(s, d, i, 3)]
            m2, m3 = a2[:, FI["rel"]] <= 96, a3[:, FI["rel"]] <= 96
            shared.append(bool(m2.sum() == m3.sum() and np.array_equal(a2[m2], a3[m3])))
    res["determinism"]["K2_K3_first_96_steps_identical"] = [int(sum(shared)), len(shared)]
    # per K
    for K in UNITS:
        full = [(s, d) for s, d in states if all((s, d, i, K) in rows for i in range(8))]
        if not full:
            continue
        T = np.array([[float(rows[(s, d, i, K)]["success"]) for i in range(8)] for s, d in full])
        nsucc = T.sum(1); mixed = (nsucc > 0) & (nsucc < 8)
        gap = T.max(1) - T.mean(1)
        out = dict(states=len(full), incomplete=[st for st in states if st not in full],
                   random=float(T.mean()), oracle=float(T.max(1).mean()), gap=float(gap.mean()), gap_ci=boot(gap),
                   mixed_states=int(mixed.sum()), all_fail=int((nsucc == 0).sum()), all_success=int((nsucc == 8).sum()),
                   success_within_unit=float(np.mean([rows[(s, d, i, K)]["success_rel"] is not None
                                                      and rows[(s, d, i, K)]["success_rel"] <= PLAN * K
                                                      for s, d in full for i in range(8)])))
        div, sep, gain = {}, {}, {}
        for rel in RELS:
            C_ = {st: [cap_at((*st, i, K), rel) for i in range(8)] for st in full}
            fp = np.array([[c[[FI["fp_x"], FI["fp_y"], FI["fp_z"]]] for c in C_[st]] for st in full])
            ee = np.array([[c[[FI["ree_x"], FI["ree_y"], FI["ree_z"]]] for c in C_[st]] for st in full])
            gr = np.array([[c[FI["grip_r"]] for c in C_[st]] for st in full])
            sd = lambda X: np.sqrt((X.std(1) ** 2).sum(-1)) * 100  # noqa: E731  cm
            div[rel] = dict(mug_fp_sd_cm_median=float(np.median(sd(fp))), mug_fp_sd_cm_median_mixed=
                            float(np.median(sd(fp)[mixed])) if mixed.any() else None,
                            right_ee_sd_cm_median=float(np.median(sd(ee))), grip_r_sd_median=float(np.median(gr.std(1))))
            Sg = np.array([[-goal_dist(c) for c in C_[st]] for st in full])
            P = np.array([[prm_at((*st, i, K), rel)[0] for i in range(8)] for st in full], dtype=object)
            has_prm = all(v is not None for v in P.ravel())
            Pp = P.astype(float) if has_prm else None
            sep[rel] = dict(goal=pair_acc(T[mixed], Sg[mixed]), prm_phi=pair_acc(T[mixed], Pp[mixed]) if has_prm else None)
            gain[rel] = dict(goal=bon(T, Sg), prm_phi=bon(T, Pp) if has_prm else None)
            # LOSO privileged (descriptive)
            groups = [dict(seed=st[0], F=np.array([priv(c) for c in C_[st]]), Y=T[j]) for j, st in enumerate(full)]
            sc = [None] * len(groups)
            for s0 in sorted({g["seed"] for g in groups}):
                tr = [g for g in groups if g["seed"] != s0]
                m = C.fit_q(tr, 0.01) if tr else None
                for j, g in enumerate(groups):
                    if g["seed"] == s0:
                        sc[j] = C.q_pred(m, g["F"]) if m else np.zeros(8)
            sep[rel]["priv_loso"] = pair_acc(T[mixed], np.array(sc)[mixed])
        summ = lambda v: None if v is None else dict(mean=float(np.mean(v)) if len(v) else None, ci=boot(v), n=len(v))  # noqa: E731
        out["divergence"] = div
        out["separability"] = {rel: {k: summ(v) for k, v in x.items()} for rel, x in sep.items()}
        out["bon_gain"] = {rel: {k: summ(v) for k, v in x.items()} for rel, x in gain.items()}
        end = PLAN * K
        g_end = gain[end]["goal"]
        out["unit_end"] = dict(rel=end, goal_gain=summ(g_end), goal_gain_per_chunk=summ(list(np.array(g_end) / K)),
                               prm_gain=summ(gain[end]["prm_phi"]) if gain[end]["prm_phi"] is not None else None,
                               goal_pair_acc=summ(sep[end]["goal"]),
                               prm_pair_acc=summ(sep[end]["prm_phi"]) if sep[end]["prm_phi"] is not None else None)
        res["per_K"][K] = out
    # recommendation (pre-registered rule)
    q = [K for K in res["per_K"] if res["per_K"][K]["unit_end"]["goal_gain"]["ci"][0] is not None
         and res["per_K"][K]["unit_end"]["goal_gain"]["ci"][0] > 0]
    rec = None
    if q:
        best = max(q, key=lambda K: res["per_K"][K]["unit_end"]["goal_gain_per_chunk"]["mean"])
        rec = best
        for K in sorted(q):
            if K < best and res["per_K"][K]["unit_end"]["goal_gain_per_chunk"]["ci"][1] >= \
                    res["per_K"][best]["unit_end"]["goal_gain_per_chunk"]["ci"][0]:
                rec = K; break
    res["recommendation"] = dict(qualifying_K=q, recommended_K=rec)
    (od / "unit_result.json").write_text(json.dumps(res, indent=1, default=str))
    lines = [f"states {len(states)}, excluded jobs {sorted(excl)}", f"determinism {res['determinism']}"]
    for K, o in res["per_K"].items():
        lines.append(f"K={K}: states {o['states']} random {o['random']:.3f} oracle {o['oracle']:.3f} gap {o['gap']:+.3f} "
                     f"CI {np.round(o['gap_ci'], 3).tolist()} mixed {o['mixed_states']} all-fail {o['all_fail']} "
                     f"all-success {o['all_success']} success-within-unit {o['success_within_unit']:.3f}")
        ue = o["unit_end"]
        lines.append(f"   unit end +{ue['rel']}: goal pair acc {ue['goal_pair_acc']} | goal gain {ue['goal_gain']} | "
                     f"per chunk {ue['goal_gain_per_chunk']} | prm pair acc {ue['prm_pair_acc']} | prm gain {ue['prm_gain']}")
        for rel in RELS:
            sp, dv = o["separability"][rel], o["divergence"][rel]
            f = lambda x: "-" if x is None or x["mean"] is None else f"{x['mean']:.3f}"  # noqa: E731
            lines.append(f"   rel {rel:3d}: mug SD {dv['mug_fp_sd_cm_median']:.2f} cm (mixed {dv['mug_fp_sd_cm_median_mixed']}) "
                         f"ee SD {dv['right_ee_sd_cm_median']:.2f} cm | pair acc goal {f(sp['goal'])} prm {f(sp['prm_phi'])} "
                         f"priv {f(sp['priv_loso'])} | gain goal {f(o['bon_gain'][rel]['goal'])} prm {f(o['bon_gain'][rel]['prm_phi'])}")
    lines.append(f"recommendation {res['recommendation']}")
    (od / "unit_result.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
