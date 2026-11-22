"""Evaluate the FROZEN gate critic on a list of held-out gate states (Claude, 2026-11-22; PROTOCOL addendum 2).
The critic is read from --critic (r0/critic_gate.json: V and Q trained on the 80 non-held-out scenes); nothing is
refit. True outcomes = one pi0 continuation per candidate (<ep>.cont.jsonl, start_ok rows only).
Primary (decision): BoN-8 gain = true outcome of the critic's top-scored candidate (first index among ties, as
critic_train.py) - mean true outcome of the 8 candidates, averaged over the listed states that have all 8 outcomes;
bootstrap over states (numpy default_rng(0), 2000 resamples, states in (seed, decision) order, as critic_train.py);
PASS iff the 2.5th percentile > 0.
Secondary (descriptive only, never used to decide): randomization p (uniform random picker, 100000 draws,
default_rng(1)); number of mixed-outcome states; gain by decision phase 0-4 / 5-9 / >=10 (phase 0-4 pre-registered
as the secondary hypothesis); gains and pair accuracies of PRM-only scorers.
usage: python gate_eval.py --round-dir r0 --scan-start 170000 --critic r0/critic_gate.json \
         --pairs-file gate_pairs_retest.json --out r0/gate_retest_result.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import critic_train as C  # noqa: E402
from prm_common import Q_FEATURES  # noqa: E402
from round_select import held_out, kept_seeds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True); ap.add_argument("--scan-start", type=int, required=True)
    ap.add_argument("--critic", type=Path, required=True); ap.add_argument("--pairs-file", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    crit = json.loads(a.critic.read_text())
    V, Q, gamma = crit["V"], crit["Q"], crit["gamma"]
    kept, _ = kept_seeds(a.round_dir, a.scan_start)
    held = set(held_out(kept))
    want = [(s["seed"], s["decision"]) for s in json.loads((a.round_dir / a.pairs_file).read_text())["states"]]
    if not all(s in held for s, _ in want):
        raise SystemExit("listed states must come from held-out scenes")
    eps, missing = C.load_round(a.round_dir, [s for s in kept if s in held])
    groups = {(g["seed"], g["d"]): g for e in eps for g in C.q_groups(e, V, gamma)}
    listed = [groups.get(st) for st in sorted(want)]
    gate = [g for g in listed if g is not None and all(t is not None for t in g["true"])]
    incomplete = [st for st, g in zip(sorted(want), listed) if g is None or not all(t is not None for t in g["true"])]
    T = np.array([g["true"] for g in gate], float)
    S = np.array([C.q_pred(Q, g["F"]) for g in gate])
    k = np.array([int(np.argmax(s)) for s in S])
    picked = T[np.arange(len(T)), k]
    gains = list(picked - T.mean(1))
    rng = np.random.default_rng(0)
    boot = [np.mean(rng.choice(gains, len(gains))) for _ in range(2000)]
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    nsucc = T.sum(1); mixed = (nsucc > 0) & (nsucc < 8)
    r1 = np.random.default_rng(1)
    null = np.array([(T[np.arange(len(T)), r1.integers(0, 8, len(T))] - T.mean(1)).mean() for _ in range(100000)])
    d = np.array([g["d"] for g in gate])
    phases = {}
    for name, lo, hi in (("0-4", 0, 4), ("5-9", 5, 9), (">=10", 10, 10 ** 6)):
        m = (d >= lo) & (d <= hi)
        if m.any():
            g_ = np.array(gains)[m]
            rb = np.random.default_rng(0)
            bt = [np.mean(rb.choice(g_, len(g_))) for _ in range(2000)]
            phases[name] = dict(states=int(m.sum()), random=float(T[m].mean()), critic=float(picked[m].mean()),
                                gain=float(g_.mean()), gain_ci=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                                mixed_states=int(mixed[m].sum()))
    fi = {n: j for j, n in enumerate(Q_FEATURES)}
    scorers = dict(q=lambda g: C.q_pred(Q, g["F"]), prm_real_phi48=lambda g: g["RE"][:, 0],
                   prm_real_hop48=lambda g: g["RE"][:, 1], prm_imag_phi48=lambda g: g["F"][:, fi["phi_im48"]],
                   prm_imag_hop48=lambda g: g["F"][:, fi["hop_cur_to_im48"]])
    res = dict(pairs_file=a.pairs_file, critic=str(a.critic), listed_states=len(want), evaluated_states=len(gate),
               incomplete_states=incomplete, missing_features=missing,
               random=float(T.mean()), picked=float(picked.mean()), oracle=float(T.max(1).mean()),
               gain=float(np.mean(gains)), gain_ci=ci, gate_pass=bool(ci[0] > 0),
               secondary=dict(randomization_p=float((null >= np.mean(gains) - 1e-12).mean()),
                              all_fail_states=int((nsucc == 0).sum()), all_success_states=int((nsucc == 8).sum()),
                              mixed_states=int(mixed.sum()),
                              critic_success_on_mixed=int(picked[mixed].sum()),
                              random_expectation_on_mixed=float(T[mixed].mean(1).sum()),
                              by_phase=phases,
                              bon_gain_by={n: C.bon_gain(gate, f) for n, f in scorers.items()},
                              pair_acc_true_by={n: C.pair_acc_true(gate, f) for n, f in scorers.items()}))
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
