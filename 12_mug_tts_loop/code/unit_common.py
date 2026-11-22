"""Shared constants of the multi-chunk unit measurement (Claude, 2026-11-22; PROTOCOL addendum 3). Pure python."""
FEATS = (["rel", "cnt", "mug_x", "mug_y", "mug_z", "mug_qw", "mug_qx", "mug_qy", "mug_qz", "fp_x", "fp_y", "fp_z",
          "mid_x", "mid_y", "mid_z"] + [f"lee_{c}" for c in ("x", "y", "z", "qw", "qx", "qy", "qz")]
         + [f"ree_{c}" for c in ("x", "y", "z", "qw", "qx", "qy", "qz")] + ["grip_l", "grip_r", "eval_success", "check_success"])
