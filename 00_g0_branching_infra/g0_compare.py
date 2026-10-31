"""Offline comparison of two g0_driver.py episode JSONs (Claude, 2026-10-27). CPU only.

Reports, per shared plan boundary (and 'final'), the maximum discrepancy per quantity, plus executed-action
equality and outcome equality. Tolerances are the consensus G0 thresholds; they are applied, never tuned.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

TOL = {"pos_m": 1e-3, "rot_rad": 1e-3, "lin_vel": 1e-3, "ang_vel": 1e-3, "qpos": 1e-4, "qvel": 1e-3,
       "drive_target": 1e-4, "gripper_val": 1e-4}


def rot_err(q1, q2):
    q1, q2 = np.asarray(q1, float), np.asarray(q2, float)
    d = abs(float(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))))
    return 2 * math.acos(min(1.0, d))


def cmp_state(a, b):
    m = {k: 0.0 for k in TOL}
    issues = []
    if set(a["actors"]) != set(b["actors"]) or set(a["articulations"]) != set(b["articulations"]):
        issues.append("entity set differs")
    for k in set(a["actors"]) & set(b["actors"]):
        x, y = a["actors"][k], b["actors"][k]
        m["pos_m"] = max(m["pos_m"], float(np.linalg.norm(np.subtract(x["p"], y["p"]))))
        m["rot_rad"] = max(m["rot_rad"], rot_err(x["q"], y["q"]))
        if "v" in x and "v" in y:
            m["lin_vel"] = max(m["lin_vel"], float(np.linalg.norm(np.subtract(x["v"], y["v"]))))
            m["ang_vel"] = max(m["ang_vel"], float(np.linalg.norm(np.subtract(x["w"], y["w"]))))
        elif ("v" in x) != ("v" in y):
            issues.append(f"velocity coverage differs: {k}")
    for k in set(a["articulations"]) & set(b["articulations"]):
        x, y = a["articulations"][k], b["articulations"][k]
        m["pos_m"] = max(m["pos_m"], float(np.linalg.norm(np.subtract(x["root_p"], y["root_p"]))))
        m["rot_rad"] = max(m["rot_rad"], rot_err(x["root_q"], y["root_q"]))
        for key, tol_key in (("qpos", "qpos"), ("qvel", "qvel")):
            if len(x[key]) != len(y[key]):
                issues.append(f"{key} length differs: {k}"); continue
            if x[key]:
                m[tol_key] = max(m[tol_key], float(np.max(np.abs(np.subtract(x[key], y[key])))))
        dt = [abs(p - q) for u, v in zip(x["drive_target"], y["drive_target"]) for p, q in zip(u, v)]
        if dt:
            m["drive_target"] = max(m["drive_target"], max(dt))
    m["gripper_val"] = float(np.max(np.abs(np.subtract(a["gripper_val"], b["gripper_val"]))))
    for key in ("action_step", "eval_success", "check_success", "np_rng_sha", "py_rng_sha"):
        if a[key] != b[key]:
            issues.append(f"{key}: {a[key]} != {b[key]}")
    for k, v in m.items():
        if not math.isfinite(v) or v >= TOL[k]:
            issues.append(f"{k} {v:.3g} >= {TOL[k]}")
    return {"max": m, "issues": issues, "pass": not issues}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--from-step", type=int, default=0, help="only compare boundaries >= this step (branch suffix)")
    a_ = ap.parse_args()
    A, B = json.loads(a_.a.read_text()), json.loads(a_.b.read_text())
    keys = sorted({k for k in A["states"] if k != "final"} & {k for k in B["states"] if k != "final"}, key=int)
    keys = [k for k in keys if int(k) >= a_.from_step]
    per = {k: cmp_state(A["states"][k], B["states"][k]) for k in keys}
    per["final"] = cmp_state(A["states"]["final"], B["states"]["final"])
    n = min(len(A["executed_actions"]), len(B["executed_actions"]))
    first_diff = next((i for i in range(n) if A["executed_actions"][i] != B["executed_actions"][i]), None)
    action_max = float(np.max(np.abs(np.subtract(A["executed_actions"][:n], B["executed_actions"][:n])))) if n else 0.0
    out = {
        "a": str(a_.a), "b": str(a_.b), "modes": [A["mode"], B["mode"]], "env_seed": [A["env_seed"], B["env_seed"]],
        "success": [A["success"], B["success"]], "steps": [A["steps"], B["steps"]],
        "actions_identical": A["executed_actions"] == B["executed_actions"],
        "first_action_diff_step": first_diff, "action_max_abs": action_max,
        "boundaries_compared": len(per), "boundaries_pass": sum(v["pass"] for v in per.values()),
        "worst": {k: max(v["max"][k] for v in per.values()) for k in TOL},
        "failing": {k: v["issues"][:4] for k, v in per.items() if not v["pass"]},
        "outcome_equal": A["success"] == B["success"] and A["steps"] == B["steps"],
    }
    out["pass"] = out["boundaries_pass"] == out["boundaries_compared"] and out["outcome_equal"] and out["actions_identical"]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
