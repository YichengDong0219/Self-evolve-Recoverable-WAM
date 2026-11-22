"""Descriptive only (11-22 13:20): how well the CURRENT Q training target Y ranks the 8 candidates' true continuation
outcomes on the 296 gate states (frozen V from critic_gate.json). Never used to choose anything."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/fact_local/mug_loop_1121/code")
import critic_train as C
from round_select import held_out, kept_seeds
RD = Path("/workspace/fact_local/mug_loop_1121/r0")
crit = json.loads((RD / "critic_gate.json").read_text())
print({k: crit[k] for k in ("gamma", "Q_l2", "Q_cv_pair_acc", "Q_train_states", "train_episodes", "V_auc_train", "V_auc_heldout")})
print("V w", np.round(crit["V"]["w"], 3), "b", round(crit["V"]["b"], 3), "(features phi, hop_prev, t_frac, grip_l, grip_r)")
print("Q w", dict(zip(crit["Q"]["features"], np.round(crit["Q"]["w"], 3))))
kept, _ = kept_seeds(RD, 170000); held = set(held_out(kept))
eps, _ = C.load_round(RD, kept)
print("episode lengths (steps):", np.percentile([e["ep"]["steps"] for e in eps], [0, 50, 100]), "step_lim", eps[0]["ep"]["step_lim"])
tr = [g for e in eps if e["seed"] not in held for g in C.q_groups(e, crit["V"], crit["gamma"])]
Ytr = np.stack([g["Y"] for g in tr])
print(f"train states {len(tr)}; target=1 (success inside the chunk) share {np.mean(Ytr == 1):.4f}")
print(f"train target spread: across-state SD {Ytr.mean(1).std():.4f}, mean within-state SD {Ytr.std(1).mean():.4f}")
want = set()
for f in ("gate_pairs.json", "gate_pairs_retest.json"):
    want |= {(s["seed"], s["decision"]) for s in json.loads((RD / f).read_text())["states"]}
gate = [g for e in eps if e["seed"] in held for g in C.q_groups(e, crit["V"], crit["gamma"]) if (g["seed"], g["d"]) in want]
print("gate states", len(gate))
print("target Y as a scorer on true outcomes: BoN gain %+.4f, pair acc %.3f" % (C.bon_gain(gate, lambda g: g["Y"]), C.pair_acc_true(gate, lambda g: g["Y"])))
print("frozen Q as a scorer:                  BoN gain %+.4f, pair acc %.3f" % (C.bon_gain(gate, lambda g: C.q_pred(crit["Q"], g["F"])), C.pair_acc_true(gate, lambda g: C.q_pred(crit["Q"], g["F"]))))
