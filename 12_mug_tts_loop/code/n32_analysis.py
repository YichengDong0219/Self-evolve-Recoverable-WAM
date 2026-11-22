"""Analysis of the N = 32 snapshot test (Claude, 2026-11-22; PROTOCOL addendum 4, written before the run).
Any python with numpy. States: r0/n32/n32_states.json (20 held-out scenes of the gate critic x decisions 0-6) minus the
excluded ones (generation audit, render-error jobs, incomplete outcome sets).
Outcomes per state, candidate ids 0-31: ids 0-7 from the gate continuations (<ep>.cont.jsonl) when the regenerated ids
0-7 equal round 0, else from the new continuations; ids 8-31 from the new continuations (<out-dir>/seed<S>.n32.jsonl,
capped at 450 steps). Blocks: ids 8b..8b+7, b = 0..3 (block 0 = the round-0 candidates).
 a. diversity: mixed share (0 < successes < N) and oracle (any success) at N = 8 (block 0, and mean over the 4 blocks)
    and N = 32; random success (mean over the 32); states with no success in block 0 but some among the 32;
 b. frozen critic (Q of r0/critic_gate.json, scores from n32_score.py): BoN gain = success of the critic's top pick
    (ties: mean over the tied) - the mean success of the pool; N = 32 (the whole pool) and N = 8 (mean over the 4
    blocks, each block its own pool); their difference; pair accuracy on the N = 32 mixed states; within-state SD of
    the scores (N = 32 vs mean block SD); by phase (decisions 0-3, 4-6), descriptive;
 c. CIs: bootstrap over the 20 scenes (cluster: all states of a resampled scene), 2000 resamples, default_rng(0);
    a state bootstrap is shown for reference;
 d. pre-registered criterion: the critic has a positive effect at N = 32 iff the scene-bootstrap 95% CI of the N = 32
    BoN gain has its lower bound > 0;
 e. checks: candidate ids 0-7 reproduction, capped continuations, successes after episode step 346, critic features of
    ids 0-7 vs the gate's stored features (PRM batches are not bit-reproducible).
usage: python n32_analysis.py --round-dir r0 --out-dir r0/n32
"""
import argparse
import json
from pathlib import Path

import numpy as np


def bon(t, s):
    top = np.isclose(s, np.max(s))
    return float(t[top].mean() - t.mean())


def pair_acc(t, s):
    ok = [1.0 if s[a] > s[b] else 0.5 if s[a] == s[b] else 0.0 for a in range(len(t)) for b in range(len(t)) if t[a] > t[b]]
    return float(np.mean(ok)) if ok else None


def boot_scene(per_state, scene_of, n=2000):
    """per_state: dict state -> value; mean over states of resampled scenes."""
    scenes = sorted({scene_of[k] for k in per_state})
    by = {s: [v for k, v in per_state.items() if scene_of[k] == s] for s in scenes}
    rng = np.random.default_rng(0)
    means = []
    for _ in range(n):
        pick = rng.choice(len(scenes), len(scenes))
        vals = [v for j in pick for v in by[scenes[j]]]
        means.append(np.mean(vals))
    return np.percentile(means, [2.5, 97.5]).round(4).tolist()


def boot_state(vals, n=2000):
    x = np.asarray(vals, float)
    rng = np.random.default_rng(0)
    return np.percentile([rng.choice(x, len(x)).mean() for _ in range(n)], [2.5, 97.5]).round(4).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    rd, od = a.round_dir, a.out_dir
    gen = od / "gen"
    states = [tuple(x) for x in json.loads((od / "n32_states.json").read_text())["states"]]
    excl = {(s, d): why for s, d, why in json.loads((od / "gen_excluded.json").read_text())}
    bad_jobs = json.loads((od / "n32_excluded_jobs.json").read_text()) if (od / "n32_excluded_jobs.json").exists() else []
    bad_pairs = {(s, d, i) for s, d, ids in bad_jobs for i in ids}
    genrow = {}
    for f in gen.glob("gen_seed*.jsonl"):
        for r in map(json.loads, f.read_text().splitlines()):
            genrow[(int(f.name[8:14]), r["decision"])] = r
    scores = {}
    if (gen / "n32_scores.jsonl").exists():
        for r in map(json.loads, (gen / "n32_scores.jsonl").read_text().splitlines()):
            scores[(r["seed"], r["decision"])] = (np.asarray(r["scores"]), np.asarray(r["feats"]))
    new, capped, late = {}, 0, 0
    for f in od.glob("seed*.n32.jsonl"):
        s = int(f.name[4:10])
        for r in map(json.loads, f.read_text().splitlines()):
            if r.get("start_ok") and "success" in r and (s, r["decision"], r["cand"]) not in bad_pairs:
                new[(s, r["decision"], r["cand"])] = float(r["success"])
                capped += bool(r.get("capped")); late += bool(r["success"] and r["steps"] > 346)
    gate = {}
    for s in {s for s, _ in states}:
        for r in map(json.loads, (rd / f"hanging_mug_seed{s}.cont.jsonl").read_text().splitlines()):
            if r.get("start_ok") and "success" in r:
                gate[(s, r["decision"], r["cand"])] = float(r["success"])
    T, S, F, missing = {}, {}, {}, []
    for s, d in states:
        if (s, d) in excl:
            continue
        reuse = bool(genrow[(s, d)].get("ids07_equal"))
        out = []
        for i in range(32):
            src = gate if (i < 8 and reuse) else new
            out.append(src.get((s, d, i)))
        if any(v is None for v in out) or (s, d) not in scores:
            missing.append([s, d, sum(v is None for v in out), (s, d) in scores]); continue
        T[(s, d)] = np.asarray(out); S[(s, d)], F[(s, d)] = scores[(s, d)]
    keys = sorted(T)
    scene_of = {k: k[0] for k in keys}
    res = dict(states_planned=len(states), excluded_generation=[[*k, v] for k, v in sorted(excl.items())],
               excluded_jobs=bad_jobs, incomplete=missing, states=len(keys), scenes=len({k[0] for k in keys}),
               ids07_reproduced=sum(bool(genrow[k].get("ids07_equal")) for k in keys), capped_continuations=capped,
               successes_after_step_346=late)
    blocks = lambda t: [t[8 * b:8 * b + 8] for b in range(4)]  # noqa: E731
    # a. diversity
    mixed8_b0 = {k: float(0 < T[k][:8].sum() < 8) for k in keys}
    mixed8_avg = {k: float(np.mean([0 < x.sum() < 8 for x in blocks(T[k])])) for k in keys}
    mixed32 = {k: float(0 < T[k].sum() < 32) for k in keys}
    orc8_b0 = {k: float(T[k][:8].max()) for k in keys}
    orc8_avg = {k: float(np.mean([x.max() for x in blocks(T[k])])) for k in keys}
    orc32 = {k: float(T[k].max()) for k in keys}
    rnd = {k: float(T[k].mean()) for k in keys}
    newly = [list(k) for k in keys if T[k][:8].sum() == 0 and T[k].sum() > 0]
    div = dict(random=np.mean(list(rnd.values())),
               mixed_share=dict(N8_block0=np.mean(list(mixed8_b0.values())), N8_mean_of_blocks=np.mean(list(mixed8_avg.values())),
                                N32=np.mean(list(mixed32.values()))),
               oracle=dict(N8_block0=np.mean(list(orc8_b0.values())), N8_mean_of_blocks=np.mean(list(orc8_avg.values())),
                           N32=np.mean(list(orc32.values())), N32_ci_scene=boot_scene(orc32, scene_of)),
               oracle_minus_random=dict(N8_mean_of_blocks=np.mean([orc8_avg[k] - rnd[k] for k in keys]),
                                        N32=np.mean([orc32[k] - rnd[k] for k in keys])),
               states_no_success_in_block0_but_some_in_32=newly,
               successes_out_of_32_histogram=np.bincount([int(T[k].sum()) for k in keys], minlength=33).tolist())
    res["diversity"] = div
    # b. critic
    g32 = {k: bon(T[k], S[k]) for k in keys}
    g8 = {k: float(np.mean([bon(t, s) for t, s in zip(blocks(T[k]), blocks(S[k]))])) for k in keys}
    diff = {k: g32[k] - g8[k] for k in keys}
    pick32 = {k: float(T[k][np.isclose(S[k], S[k].max())].mean()) for k in keys}
    pacc = [x for x in (pair_acc(T[k], S[k]) for k in keys if mixed32[k]) if x is not None]
    sd32 = np.mean([S[k].std() for k in keys]); sd8 = np.mean([np.mean([s.std() for s in blocks(S[k])]) for k in keys])
    crit = dict(bon_gain_N32=np.mean(list(g32.values())), bon_gain_N32_ci_scene=boot_scene(g32, scene_of),
                bon_gain_N32_ci_state=boot_state(list(g32.values())),
                bon_gain_N8=np.mean(list(g8.values())), bon_gain_N8_ci_scene=boot_scene(g8, scene_of),
                N32_minus_N8=np.mean(list(diff.values())), N32_minus_N8_ci_scene=boot_scene(diff, scene_of),
                critic_pick_success_N32=np.mean(list(pick32.values())), random_success=div["random"],
                pair_acc_mixed_N32=dict(mean=float(np.mean(pacc)) if pacc else None, n=len(pacc),
                                        ci_state=boot_state(pacc) if pacc else None),
                score_sd_within_state=dict(N32=float(sd32), N8_mean_of_blocks=float(sd8)))
    ph = {}
    for name, lo, hi in (("decisions_0_3", 0, 3), ("decisions_4_6", 4, 6)):
        kk = [k for k in keys if lo <= k[1] <= hi]
        if kk:
            ph[name] = dict(states=len(kk), mixed32=float(np.mean([mixed32[k] for k in kk])),
                            oracle32_minus_random=float(np.mean([orc32[k] - rnd[k] for k in kk])),
                            bon_gain_N32=float(np.mean([g32[k] for k in kk])), bon_gain_N8=float(np.mean([g8[k] for k in kk])))
    crit["by_phase"] = ph
    crit["positive_effect_N32"] = bool(crit["bon_gain_N32_ci_scene"][0] > 0)
    res["critic"] = crit
    # e. checks: features of ids 0-7 vs the gate's stored features
    fd = []
    for k in keys:
        if not genrow[k].get("ids07_equal"):
            continue
        p = json.loads((rd / f"hanging_mug_seed{k[0]}.prm.json").read_text())["features"]
        for i in range(8):
            key = f"{k[1]}_{i}"
            if key in p.get("phi_im48", {}):
                fd.append(np.abs(np.asarray(F[k][i][:3]) - np.asarray([p["hop_cur_to_im48"][key], p["phi_im48"][key], p["phi_im24"][key]])))
    if fd:
        fd = np.asarray(fd)
        res["checks_ids07_prm_features_vs_gate"] = dict(n=len(fd), mean_abs_diff=fd.mean(0).round(4).tolist(),
                                                         share_identical=float((fd.max(1) == 0).mean()))
    js = lambda o: json.loads(json.dumps(o, default=float))  # noqa: E731
    (od / "n32_result.json").write_text(json.dumps(js(res), indent=1))
    lines = [f"states {res['states']} ({res['scenes']} scenes) of {len(states)}; excluded generation {len(excl)}, "
             f"render-error jobs {len(bad_jobs)}, incomplete {len(missing)}; ids 0-7 reproduced in {res['ids07_reproduced']}; "
             f"capped continuations {capped}; successes after step 346: {late}",
             f"diversity: random {div['random']:.3f} | mixed share N8 block0 {div['mixed_share']['N8_block0']:.3f}, "
             f"N8 mean of blocks {div['mixed_share']['N8_mean_of_blocks']:.3f}, N32 {div['mixed_share']['N32']:.3f} | "
             f"oracle N8 block0 {div['oracle']['N8_block0']:.3f}, N8 mean {div['oracle']['N8_mean_of_blocks']:.3f}, "
             f"N32 {div['oracle']['N32']:.3f} {div['oracle']['N32_ci_scene']} | oracle-random N8 "
             f"{div['oracle_minus_random']['N8_mean_of_blocks']:+.3f}, N32 {div['oracle_minus_random']['N32']:+.3f}",
             f"   states with no success in block 0 but some among the 32: {len(newly)} {newly}",
             f"   successes out of 32 (histogram 0..32): {div['successes_out_of_32_histogram']}",
             f"critic: BoN gain N32 {crit['bon_gain_N32']:+.4f} scene CI {crit['bon_gain_N32_ci_scene']} (state CI "
             f"{crit['bon_gain_N32_ci_state']}) | N8 {crit['bon_gain_N8']:+.4f} {crit['bon_gain_N8_ci_scene']} | "
             f"N32-N8 {crit['N32_minus_N8']:+.4f} {crit['N32_minus_N8_ci_scene']}",
             f"   critic pick success N32 {crit['critic_pick_success_N32']:.3f} vs random {crit['random_success']:.3f}; pair acc "
             f"(N32 mixed) {crit['pair_acc_mixed_N32']}; score SD within state N32 {crit['score_sd_within_state']['N32']:.4f} "
             f"vs N8 {crit['score_sd_within_state']['N8_mean_of_blocks']:.4f}",
             f"   by phase: {crit['by_phase']}",
             f"   checks: {res.get('checks_ids07_prm_features_vs_gate')}",
             f"PRE-REGISTERED: positive effect at N=32 -> {crit['positive_effect_N32']}"]
    (od / "n32_result.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
