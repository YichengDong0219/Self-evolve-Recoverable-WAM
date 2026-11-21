"""ATC G1b helper (Claude, 2026-11-21). CPU only.
  check  compare the step-0 x0 value of a G1b run with G1's sigma = 1 value on the same chunks (protocol gate: r >= 0.99)
  merge  write an npz in the probe-script format (side, value_pred = G1b denoised value, feat_* copied from G1) so that
         atc_probe_cpu.py evaluates the denoised value exactly like G1 and R2
usage: python g1b_check_and_merge.py check <g1b npz> <g1 npz>
       python g1b_check_and_merge.py merge <g1b npz> <g1 npz> <out npz>
"""
import json
import sys

import numpy as np

mode, b_path, g_path = sys.argv[1], sys.argv[2], sys.argv[3]
b, g = np.load(b_path), np.load(g_path)
sb, sg = [str(x) for x in b["side"]], [str(x) for x in g["side"]]
assert sg[:len(sb)] == sb, "chunk order differs"
n = len(sb)
step0 = b["value_step0"].astype(np.float64); fin = b["value_final"].astype(np.float64)
g1 = g["value_pred"][:n].astype(np.float64)
if mode == "check":
    r = float(np.corrcoef(step0.ravel(), g1.ravel())[0, 1])
    r_mean = float(np.corrcoef(step0.mean(1), g1.mean(1))[0, 1]) if n > 2 else None
    print(json.dumps(dict(chunks=n, r_step0_vs_g1_per_draw=round(r, 4), r_step0_vs_g1_mean=r_mean and round(r_mean, 4),
                          max_abs_diff=round(float(np.abs(step0 - g1).max()), 4),
                          final_range=[round(float(fin.min()), 3), round(float(fin.max()), 3)],
                          final_vs_step0_r=round(float(np.corrcoef(fin.ravel(), step0.ravel())[0, 1]), 4),
                          gate_pass=bool(r >= 0.99))))
elif mode == "merge":
    assert n == len(sg), "G1b must cover all chunks"
    np.savez_compressed(sys.argv[4], side=b["side"], value_pred=fin.astype(np.float32),
                        **{k: g[k] for k in ("feat_fs", "feat_val", "feat_img")})
    print("MERGED", sys.argv[4])
else:
    raise SystemExit(mode)
