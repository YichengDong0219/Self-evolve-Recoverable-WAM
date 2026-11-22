"""Synthetic TTS round for testing critic_train.py / round_select.py / gate_select.py (no real data).
usage: python synth_round.py <out dir> <signal 0..1> [seed]
signal = how strongly the true continuation outcome depends on the candidate's PRM features."""
import json
import sys
from pathlib import Path

import numpy as np

out, signal = Path(sys.argv[1]), float(sys.argv[2])
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
out.mkdir(parents=True, exist_ok=True)
START, N, LIM = 170000, 8, 900
for s in range(START, START + 140):
    if rng.random() < 0.2:
        (out / f"hanging_mug_seed{s}.invalid").touch(); continue
    succ = rng.random() < 0.35
    steps = int(rng.integers(250, 400)) if succ else LIM
    D = (steps + 47) // 48
    cap = lambda: {"gripper_val": [float(rng.random()), float(rng.random())]}  # noqa: E731
    ep = {"env_seed": s, "success": succ, "steps": steps, "step_lim": LIM, "instruction": "hang the mug",
          "states": {str(48 * d): cap() for d in range(D)}}
    (out / f"hanging_mug_seed{s}.json").write_text(json.dumps(ep))
    np.savez(out / f"hanging_mug_seed{s}.tts.npz", cands=rng.normal(size=(D, N, 48, 14)).astype(np.float32),
             state=rng.normal(size=(D, 14)).astype(np.float32), k=rng.integers(0, N, D))
    feats = {k: {} for k in ["phi_cur", "hop_prev", "hop_cur_to_im48", "phi_im48", "phi_im24",
                             "hop_cur_to_re48", "phi_re48", "phi_re24"]}
    rows, cont = [], []
    for d in range(D):
        prog = d / max(D - 1, 1) * (0.8 if succ else 0.4)
        feats["phi_cur"][str(d)] = round(prog + 0.1 * rng.normal(), 2)
        if d:
            feats["hop_prev"][str(d)] = round(0.1 * rng.normal(), 2)
        for i in range(N):
            q = rng.normal()                       # latent candidate quality
            key = f"{d}_{i}"
            feats["phi_re48"][key] = round(prog + 0.1 * q + 0.05 * rng.normal(), 2)
            feats["hop_cur_to_re48"][key] = round(0.1 * q + 0.05 * rng.normal(), 2)
            feats["phi_re24"][key] = round(prog + 0.05 * q + 0.05 * rng.normal(), 2)
            feats["phi_im48"][key] = round(prog + 0.1 * q + 0.1 * rng.normal(), 2)
            feats["hop_cur_to_im48"][key] = round(0.1 * q + 0.1 * rng.normal(), 2)
            feats["phi_im24"][key] = round(prog + 0.05 * q + 0.1 * rng.normal(), 2)
            swc = bool(succ and d == D - 1 and rng.random() < 0.3)
            rows.append(dict(decision=d, cand=i, start_ok=True, success_within_chunk=swc, end_capture=cap()))
            p_true = 1 / (1 + np.exp(-(signal * 2.5 * q + (1.0 if succ else -1.0))))
            cont.append(dict(decision=d, cand=i, start_ok=True, success=bool(rng.random() < p_true)))
    (out / f"hanging_mug_seed{s}.exec.json").write_text(json.dumps(dict(rows=rows)))
    (out / f"hanging_mug_seed{s}.prm.json").write_text(json.dumps(dict(features=feats)))
    (out / f"hanging_mug_seed{s}.cont_all.jsonl").write_text("\n".join(json.dumps(r) for r in cont) + "\n")
print("done", out)
