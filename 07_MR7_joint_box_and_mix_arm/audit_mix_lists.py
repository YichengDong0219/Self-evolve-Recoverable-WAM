"""Train-readiness audit of a hanging_mug SFT list dir (Claude, 2026-10-30). CPU only, read-only.

usage: python audit_mix_lists.py <list dir> <norm_stats_delta.json> [out.json]
Reference distribution = the stock expert demos of the E arm (sft_lists_final_g7_1029/E_expert.jsonl, 100 seeds), i.e.
what the E arm trained on without problems. Per recovery row (R_recovery + R_prefix) it checks:
  lists      seeds of R_recovery == R_prefix, disjoint from E_expert, union == the 100 E-arm seeds; no duplicate npz;
             instruction identical to the same-seed expert demo
  integrity  arrays finite, rgb/state lengths equal, no all-black frame, no frozen video while the state moves
  takeover   recovery frame 0 == FACT's last action (the drive target the recovery starts from; prefix_state[-1] is
             one frame earlier); prefix end/onset inside the prefix; the mug really
             fell (z drop) and is at rest at `end`
  joints     G7 max single-frame arm step, G8 asset range, MR7 box, max |z| of the state under FACT's stats
  actions    FACT delta chunks (state[t+k] - state[t], k=1..48) on the action-supervised frames, max |z| under FACT's
             action stats vs the reference expert maximum
  gripper    number of close/open events per gripper (expected: one close + one open on the recovering arm)
  motion     settle frames (action_masked_until), longest stall after the first motion, idle arm path length
  gates      G1-G8 of the selected attempt as recorded by sim_select
"""
import json
import os
import sys

import numpy as np

L = "/workspace/fact_local"
E_LIST = f"{L}/sft_lists_final_g7_1029"
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
LEFT, RIGHT = [0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]
GRIP = {"left": 6, "right": 13}
CHUNK = 48


def rd(p):
    return [json.loads(x) for x in open(p)]


def chunk_z(q, start, am, astd):
    """max |z| of FACT delta-action chunks starting at frames >= start (action[t] = state[t+1])"""
    zmax = 0.0
    for t in range(start, len(q) - 1):
        tgt = q[t + 1:t + 1 + CHUNK]
        z = np.abs((tgt - q[t] - am) / astd)[:, ARM]
        zmax = max(zmax, float(z.max()))
    return zmax


def grip_events(g, thr=0.02):
    """close/open events of a gripper trace (drive target), ignoring changes below thr"""
    d = np.diff(g)
    ev, cur = [], 0
    for x in d:
        s = 1 if x > 1e-4 else (-1 if x < -1e-4 else 0)
        if s and s != cur:
            ev.append("open" if s > 0 else "close")
        cur = s if s else cur
    # merge an event only if the net change is significant
    return ev, float(g.max() - g.min()) > thr


def longest_stall(q, start, eps=1e-4):
    mv = np.abs(np.diff(q, axis=0)).max(1) > eps
    best = run = 0
    for m in mv[start:]:
        run = 0 if m else run + 1
        best = max(best, run)
    return best


def main(ld, ns_path, out=None):
    ns = json.load(open(ns_path))["norm_stats"]
    sm, ss = np.array(ns["observation.state"]["mean"]), np.array(ns["observation.state"]["std"])
    am, astd = np.array(ns["action"]["mean"]), np.array(ns["action"]["std"])
    meta = json.load(open(f"{L}/assets/rf_hanging_mug_v1/meta.json"))
    g8lo, g8hi = np.array(meta["G8_range"]["lo"]), np.array(meta["G8_range"]["hi"])
    box = json.load(open(f"{L}/code/mr7_joint_box.json"))
    blo = np.array(box["left"]["lo"] + [-9] + box["right"]["lo"] + [-9])
    bhi = np.array(box["left"]["hi"] + [9] + box["right"]["hi"] + [9])
    R, Pf, E = rd(f"{ld}/R_recovery.jsonl"), rd(f"{ld}/R_prefix.jsonl"), rd(f"{ld}/E_expert.jsonl")
    mix = json.load(open(f"{ld}/mix.json")) if os.path.exists(f"{ld}/mix.json") else {"pool": {}}
    Eall = {r["env_seed"]: r for r in rd(f"{E_LIST}/E_expert.jsonl")}
    issues = []

    # lists
    rs, ps, es = [r["env_seed"] for r in R], [r["env_seed"] for r in Pf], [r["env_seed"] for r in E]
    lists = {"n_recovery": len(R), "n_expert": len(E), "recovery_eq_prefix": rs == ps,
             "disjoint": not (set(rs) & set(es)), "union_is_E100": set(rs) | set(es) == set(Eall),
             "dup_npz": len({r["npz"] for r in R}) != len(R),
             "prefix_npz_eq_recovery_npz": all(a["npz"] == b["npz"] for a, b in zip(R, Pf)),
             "instruction_eq_expert": all(r["instruction"] == Eall[r["env_seed"]]["instruction"] for r in R)}
    for k, v in lists.items():
        if (k == "dup_npz" and v) or (k not in ("dup_npz", "n_recovery", "n_expert") and not v):
            issues.append(("lists", k))

    # reference: E-arm expert demos
    ref = {"step": [], "z_state": [], "z_chunk": [], "len": [], "grip": []}
    for s, e in Eall.items():
        q = np.load(e["npz"])["expert_state"]
        ref["step"].append(float(np.abs(np.diff(q[:, ARM], axis=0)).max()))
        ref["z_state"].append(float(np.abs((q - sm) / ss)[:, ARM].max()))
        ref["z_chunk"].append(chunk_z(q, 0, am, astd))
        ref["len"].append(len(q))
    refs = {k: {"min": round(min(v), 3), "median": round(float(np.median(v)), 3), "max": round(max(v), 3)}
            for k, v in ref.items() if v}

    rows = []
    for r, p in zip(R, Pf):
        s = r["env_seed"]
        z = np.load(r["npz"])
        q, X = z["expert_state"], z["expert_rgb"]
        Q, PX, mug, QA = z["prefix_state"], z["prefix_rgb"], z["prefix_mug_p"], z["prefix_action"]
        info = mix["pool"].get(str(s), {})
        arm = info.get("arm", "right")
        idle = LEFT if arm == "right" else RIGHT
        work = RIGHT if arm == "right" else LEFT
        am_ = r["action_masked_until"]
        end, on = p["end"], p["failure_active_from"]
        row = {"seed": s, "source": info.get("source"), "flip": info.get("flip"), "frames": len(q),
               "settle": am_}
        # integrity
        row["finite"] = bool(all(np.isfinite(z[k]).all() for k in ("expert_state", "prefix_state", "prefix_mug_p")))
        row["len_match"] = len(X) == len(q) and len(PX) == len(Q) == len(mug)
        means = X.reshape(len(X), -1).mean(1)
        row["black_frames"] = int((means < 3).sum())
        fd = np.abs(np.diff(X[:, 0].astype(np.int16), axis=0)).mean((1, 2, 3))
        moving = np.abs(np.diff(q, axis=0)).max(1) > 1e-3
        row["frozen_video_while_moving"] = int(((fd < 0.05) & moving).sum())
        # takeover continuity
        row["state_gap_takeover"] = round(float(np.abs(q[0] - QA[-1])[ARM].max()), 4)
        row["img_gap_takeover"] = round(float(np.abs(X[0, 0].astype(np.int16) - PX[-1, 0].astype(np.int16)).mean()), 2)
        row["prefix_len"], row["onset"], row["end"] = len(Q), on, end
        row["prefix_idx_ok"] = 0 <= on <= end < len(Q)
        row["mug_drop_cm"] = round(float((mug[:on + 1, 2].max() - mug[end, 2]) * 100), 1)
        row["mug_rest_mm"] = round(float(np.linalg.norm(mug[min(end + 10, len(mug) - 1)] - mug[end]) * 1000), 2)
        # joints
        row["max_step"] = round(float(np.abs(np.diff(q[:, ARM], axis=0)).max()), 4)
        row["G8_oor_frames"] = int(((q < g8lo) | (q > g8hi))[:, ARM].any(1).sum())
        row["box_oor_frames"] = int(((q < blo - 1e-3) | (q > bhi + 1e-3))[:, ARM].any(1).sum())
        row["z_state_max"] = round(float(np.abs((q - sm) / ss)[:, ARM].max()), 2)
        # actions (supervised frames only)
        row["z_chunk_max"] = round(chunk_z(q, am_, am, astd), 2)
        # gripper
        ev_w, _ = grip_events(q[:, GRIP[arm]])
        ev_i, idle_grip_moves = grip_events(q[:, GRIP["right" if arm == "left" else "left"]])
        row["grip_events_work"] = ev_w
        row["grip_events_idle"] = ev_i
        # motion
        row["stall_after_first_motion"] = longest_stall(q, am_)
        row["idle_arm_path"] = round(float(np.abs(np.diff(q[:, idle], axis=0)).sum()), 3)
        row["work_arm_path"] = round(float(np.abs(np.diff(q[:, work], axis=0)).sum()), 3)
        # gates
        sel = json.load(open(info["selection"])) if info.get("selection") else {}
        t = next((x for x in sel.get("tried", []) if x.get("pass")), {})
        row["gates_fail"] = [k for k, v in t.get("gates", {}).items() if not v]
        row["gates_n"] = len(t.get("gates", {}))
        # verdicts
        bad = []
        if not (row["finite"] and row["len_match"]): bad.append("integrity")
        if row["black_frames"] or row["frozen_video_while_moving"] > 5: bad.append("video")
        if row["state_gap_takeover"] > 1e-3: bad.append("takeover_state_gap")
        if not row["prefix_idx_ok"]: bad.append("prefix_index")
        if row["mug_drop_cm"] < 2: bad.append("no_real_fall")
        if row["mug_rest_mm"] > 1.5: bad.append("prefix_end_not_at_rest")
        if row["max_step"] > 0.2: bad.append("G7")
        if row["G8_oor_frames"]: bad.append("G8")
        if row["z_state_max"] > refs["z_state"]["max"] + 0.5: bad.append("state_z_above_expert")
        if row["z_chunk_max"] > refs["z_chunk"]["max"] + 0.5: bad.append("chunk_z_above_expert")
        if ev_w.count("close") != 1 or ev_w.count("open") < 1: bad.append("gripper_work")
        if idle_grip_moves: bad.append("idle_gripper_moves")
        if row["gates_fail"]: bad.append("gates")
        row["flags"] = bad
        rows.append(row)
    summ = {"lists": lists, "list_issues": issues, "reference_expert": refs,
            "n_clean": sum(not r["flags"] for r in rows),
            "flag_counts": {f: sum(f in r["flags"] for r in rows) for f in sorted({f for r in rows for f in r["flags"]})}}
    for k in ("max_step", "z_state_max", "z_chunk_max", "settle", "stall_after_first_motion", "idle_arm_path",
              "mug_drop_cm", "img_gap_takeover", "box_oor_frames", "frames"):
        v = [r[k] for r in rows]
        summ[k] = {"min": min(v), "median": float(np.median(v)), "max": max(v)}
    print(json.dumps(summ, indent=1))
    for r in rows:
        if r["flags"]:
            print("FLAG", r["seed"], r["source"], r["flags"], {k: r[k] for k in (
                "max_step", "z_state_max", "z_chunk_max", "grip_events_work", "grip_events_idle", "mug_drop_cm",
                "mug_rest_mm", "state_gap_takeover", "stall_after_first_motion", "frozen_video_while_moving")})
    if out:
        json.dump({"summary": summ, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
