"""Distillation-readiness audit of MR8 kept trajectories (Claude, 2026-11-01). CPU only, read-only.

  python audit_mr8.py ref <E_expert.jsonl> <norm_stats_delta.json> <out ref.json>      (where the expert npz are)
  python audit_mr8.py audit <run dir> <ref.json> <norm_stats_delta.json> [out.json]     (where the run is)

Same checks as the round-1 audit (code/audit_mix_lists.py), on the selected attempt of every state, plus the
"rotation" problems seen before (10-30: wrist roll up to 6.9 rad, base yaw 5.2 rad, 56/100 recoveries):
  joints     max single-frame arm step (G7), frames outside the asset G8 range and how far (rad), |q| > pi frames,
             per-joint value range vs the expert demos' range, max |z| of the state under FACT's stats
  rotation   per arm joint: cumulative rotation (sum |dq|) and net rotation over the recovery vs the expert maximum
  actions    FACT delta chunks (48 steps) on action-supervised frames, max |z| vs the expert maximum
  takeover   recovery frame 0 == FACT's last executed action
  integrity  finite arrays, rgb/state lengths, black or frozen frames
  motion     settle frames (before the first arm motion), longest stall after it, gripper events per arm
Reference = the 100 stock expert demos the round-1 expert arm trained on.
"""
import glob
import json
import os
import sys

import numpy as np

ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
NAMES = [f"L{j + 1}" for j in range(6)] + [f"R{j + 1}" for j in range(6)]
GRIP = {"left": 6, "right": 13}
CHUNK = 48


def stats(ns_path):
    ns = json.load(open(ns_path))["norm_stats"]
    return (np.array(ns["observation.state"]["mean"]), np.array(ns["observation.state"]["std"]),
            np.array(ns["action"]["mean"]), np.array(ns["action"]["std"]))


def chunk_z(q, start, am, astd):
    zmax = 0.0
    for t in range(start, len(q) - 1):
        z = np.abs((q[t + 1:t + 1 + CHUNK] - q[t] - am) / astd)[:, ARM]
        zmax = max(zmax, float(z.max()))
    return zmax


def first_motion(q, eps=1e-3):
    mv = np.where(np.abs(np.diff(q[:, ARM], axis=0)).max(1) > eps)[0]
    return int(mv[0]) if len(mv) else len(q) - 1


def longest_stall(q, start, eps=1e-4):
    mv = np.abs(np.diff(q, axis=0)).max(1) > eps
    best = run = 0
    for m in mv[start:]:
        run = 0 if m else run + 1
        best = max(best, run)
    return best


def grip_events(g):
    ev, cur = [], 0
    for x in np.diff(g):
        s = 1 if x > 1e-3 else (-1 if x < -1e-3 else 0)
        if s and s != cur:
            ev.append("open" if s > 0 else "close")
        cur = s if s else cur
    return ev


def make_ref(e_list, ns_path, out):
    sm, ss, am, astd = stats(ns_path)
    R = {"step": [], "z_state": [], "z_chunk": [], "len": [], "cum": [], "net": [], "lo": [], "hi": [], "absmax": []}
    for line in open(e_list):
        q = np.load(json.loads(line)["npz"])["expert_state"]
        R["step"].append(float(np.abs(np.diff(q[:, ARM], axis=0)).max()))
        R["z_state"].append(float(np.abs((q - sm) / ss)[:, ARM].max()))
        R["z_chunk"].append(chunk_z(q, 0, am, astd))
        R["len"].append(len(q))
        R["cum"].append(np.abs(np.diff(q[:, ARM], axis=0)).sum(0).tolist())
        R["net"].append(np.abs(q[-1, ARM] - q[0, ARM]).tolist())
        R["lo"].append(q[:, ARM].min(0).tolist())
        R["hi"].append(q[:, ARM].max(0).tolist())
        R["absmax"].append(np.abs(q[:, ARM]).max(0).tolist())
    ref = {"n": len(R["len"]),
           "step_max": max(R["step"]), "z_state_max": max(R["z_state"]), "z_chunk_max": max(R["z_chunk"]),
           "len_median": float(np.median(R["len"])),
           "cum_max": np.max(R["cum"], 0).tolist(), "cum_median": np.median(R["cum"], 0).tolist(),
           "net_max": np.max(R["net"], 0).tolist(),
           "lo": np.min(R["lo"], 0).tolist(), "hi": np.max(R["hi"], 0).tolist(),
           "absmax": np.max(R["absmax"], 0).tolist()}
    json.dump(ref, open(out, "w"), indent=1)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in ref.items()
                      if not isinstance(v, list)}))


def audit(run, ref_path, ns_path, out=None):
    ref = json.load(open(ref_path))
    sm, ss, am, astd = stats(ns_path)
    here = os.path.dirname(os.path.abspath(__file__))
    meta = json.load(open(os.path.join(os.path.dirname(here), "assets/rf_hanging_mug_v1/meta.json")))
    g8lo, g8hi = np.array(meta["G8_range"]["lo"]), np.array(meta["G8_range"]["hi"])
    elo, ehi = np.array(ref["lo"]), np.array(ref["hi"])
    rows = []
    for sp in sorted(glob.glob(os.path.join(run, "*", "selection.json"))):
        S = json.load(open(sp))
        sel = S.get("selected")
        if not sel:
            continue
        r = json.load(open(sel["json"]))
        z = np.load(sel["json"][:-5] + ".npz")
        q, X = z["expert_state"], z["expert_rgb"]
        QA = z["prefix_action"]
        qa = q[:, ARM]
        fm = first_motion(q)
        row = {"state": os.path.basename(os.path.dirname(sp)), "path": "hook" if sel.get("hook") else "restage",
               "frames": len(q), "prefix": len(QA), "total": len(q) + len(QA), "settle": fm}
        # integrity
        row["finite"] = bool(all(np.isfinite(z[k]).all() for k in ("expert_state", "prefix_state")))
        row["len_match"] = len(X) == len(q)
        row["black_frames"] = int((X.reshape(len(X), -1).mean(1) < 3).sum())
        fd = np.abs(np.diff(X[:, 0].astype(np.int16), axis=0)).mean((1, 2, 3))
        row["frozen_while_moving"] = int(((fd < 0.05) & (np.abs(np.diff(q, axis=0)).max(1) > 1e-3)).sum())
        row["takeover_gap"] = round(float(np.abs(q[0] - QA[-1])[ARM].max()), 4)
        # joints / rotation
        row["max_step"] = round(float(np.abs(np.diff(qa, axis=0)).max()), 3)
        oor = np.maximum(g8lo[ARM] - qa, 0) + np.maximum(qa - g8hi[ARM], 0)
        row["G8_oor_frames"] = int((oor > 0).any(1).sum())
        row["G8_oor_max_rad"] = round(float(oor.max()), 3)
        row["G8_oor_joints"] = sorted({NAMES[j] for j in np.where((oor > 0).any(0))[0]})
        eo = np.maximum(elo - qa, 0) + np.maximum(qa - ehi, 0)
        row["expert_range_oor_max_rad"] = round(float(eo.max()), 3)
        row["expert_range_oor_joints"] = {NAMES[j]: round(float(eo[:, j].max()), 3) for j in np.where(eo.max(0) > 0)[0]}
        row["abs_gt_pi_frames"] = int((np.abs(qa) > np.pi).any(1).sum())
        row["absmax"] = round(float(np.abs(qa).max()), 3)
        cum = np.abs(np.diff(qa, axis=0)).sum(0)
        row["cum_rot_over_expert_max"] = {NAMES[j]: round(float(cum[j] / ref["cum_max"][j]), 2) for j in range(12)
                                          if cum[j] > ref["cum_max"][j]}
        row["z_state_max"] = round(float(np.abs((q - sm) / ss)[:, ARM].max()), 2)
        row["z_chunk_max"] = round(chunk_z(q, fm, am, astd), 2)
        # motion / gripper
        row["stall_after_first_motion"] = longest_stall(q, fm)
        row["grip_left"] = grip_events(q[:, GRIP["left"]])
        row["grip_right"] = grip_events(q[:, GRIP["right"]])
        hang = next((e for e in r.get("mr_log") or [] if e["stage"] == "hang"), {})
        row["expert"] = hang.get("expert")
        row["final_retreat_ok"] = bool(r.get("plan_success"))
        # verdict (round-1 thresholds; G8 is required at selection time since 11-01, so it is not re-flagged here)
        bad = []
        if not (row["finite"] and row["len_match"]) or row["black_frames"] or row["frozen_while_moving"] > 5:
            bad.append("integrity/video")
        if row["takeover_gap"] > 1e-3: bad.append("takeover_gap")
        if row["max_step"] > 0.2: bad.append("G7")
        if row["z_state_max"] > ref["z_state_max"] + 0.5: bad.append("state_z_above_expert")
        if row["z_chunk_max"] > ref["z_chunk_max"] + 0.5: bad.append("chunk_z_above_expert")
        if row["abs_gt_pi_frames"]: bad.append("joint_beyond_pi")
        if row["total"] > 900: bad.append("L900")
        if not row["final_retreat_ok"]: bad.append("no_final_retreat")
        row["flags"] = bad
        rows.append(row)
    summ = {"ref": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ref.items() if not isinstance(v, list)},
            "n": len(rows), "n_clean": sum(not r["flags"] for r in rows),
            "flag_counts": {f: sum(f in r["flags"] for r in rows) for f in sorted({f for r in rows for f in r["flags"]})}}
    res = {"summary": summ, "rows": rows}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if out:
        json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    (make_ref if sys.argv[1] == "ref" else audit)(*sys.argv[2:])
