"""Build the reusable "recoverable failures" asset for hanging_mug from the MR6 collection (Claude, 2026-10-30).

One row per FACT failure state with a takeover step (bench/stream_jobs_v7.jsonl), joined with the MR6 simulate-and-
select verdicts (bench/SEL_v7/*/selection.json), the captured recovery (npz), the same-seed expert demo and the frozen
SFT list. Rollout files are hard-linked into the asset (no extra disk; survives deletion of the originals); the
asset is written once into a new directory and made read-only. Planner results of later iterations go elsewhere,
keyed by state_id (see README.md).
usage: python build_rf_assets.py <out_dir>
"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

L = "/workspace/fact_local"
SEL = f"{L}/bench/SEL_v7"
JOBS = f"{L}/bench/stream_jobs_v7.jsonl"
FINAL = f"{L}/sft_lists_final_g7_1029"
EXPERT = f"{L}/expert_mug"
G7_MAX = 0.2   # rad, single-frame arm-joint step (mug_sft_lists.G7_MAX_STEP)
MARGIN = 0.5   # rad, G8 range margin (as in the 10-30 audit)
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def absp(p):
    """Some selection/capture paths were written relative to code/ (sim_select ran from there)."""
    return p if p is None or os.path.isabs(p) else os.path.normpath(os.path.join(L, "code", p))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(out):
    assert not os.path.exists(out), f"{out} exists; the asset is write-once"
    for sub in ("rollouts", "recoveries", "expert"):
        os.makedirs(f"{out}/{sub}")

    def keep(src, dst):
        """hard link into the asset (no extra disk) so later clean-ups of bench/ cannot break the asset"""
        if src and os.path.exists(src) and not os.path.exists(f"{out}/{dst}"):
            os.link(src, f"{out}/{dst}")
            return dst
        return dst if src and os.path.exists(f"{out}/{dst}") else None
    jobs = [json.loads(x) for x in open(JOBS)]
    final_npz = {absp(json.loads(x)["npz"]) for x in open(f"{FINAL}/R_recovery.jsonl")}
    # G8 reference range: hanging_mug expert demos of the frozen list + FACT failure prefixes
    E = np.concatenate([np.load(json.loads(x)["npz"])["expert_state"] for x in open(f"{FINAL}/E_expert.jsonl")])
    P = np.concatenate([np.load(n)["prefix_state"] for n in final_npz])
    lo, hi = np.minimum(E.min(0), P.min(0)) - MARGIN, np.maximum(E.max(0), P.max(0)) + MARGIN
    rows, seen = [], set()
    for j in jobs:
        roll = j["rollout"]
        r = json.load(open(roll))
        seed = r["env_seed"]
        sid = f"hanging_mug/{seed}/step{j['step']}"
        if sid in seen:
            continue
        seen.add(sid)
        link = f"rollouts/{os.path.basename(roll)}"
        if not os.path.exists(f"{out}/{link}"):
            os.link(roll, f"{out}/{link}")
        row = {"state_id": sid, "task": "hanging_mug", "config": r["config"], "env_seed": seed,
               "instruction": r["instruction"], "rollout": link, "rollout_orig": roll, "rollout_sha256": sha(roll),
               "takeover_step": j["step"], "failure_type": j.get("onset"), "policy_steps": r["steps"]}
        d = f"{SEL}/{j['group']}_{seed}"
        s = json.load(open(f"{d}/selection.json")) if os.path.exists(f"{d}/selection.json") else None
        if s is not None and s.get("step") != j["step"]:
            s = None  # verdict belongs to another takeover step of the same seed
        if s is None or "error" in s:
            tier = "U_untried"
            mr6 = None
        else:
            cands = []
            for vi in (0, 1):
                p = f"{d}/list_v{vi}.json"
                if os.path.exists(p):
                    lj = json.load(open(p))
                    ch = next((e for e in lj.get("mr_log", []) if e.get("stage") == "chain"), {})
                    cands += [{"variant": lj.get("variant"), **f} for f in ch.get("feasible", [])]
            sel = s.get("selected")
            if sel:
                sel = {**sel, "json": absp(sel["json"]), "video": absp(sel["video"])}
            mr6 = {"n_candidates": s["n_candidates"], "candidates": cands, "n_attempts": len(s["tried"]),
                   "attempts": [{k: t.get(k) for k in ("variant", "arm", "cid", "status", "layer", "gates", "pass")}
                                for t in s["tried"]],
                   "attempt_files": sorted(os.path.basename(x) for x in glob.glob(f"{d}/cand_*.json")),
                   "dir": d, "selected": None}
            if sel:
                tier = "A_recovered"
                cj = json.load(open(sel["json"]))
                cap = dict(cj.get("capture") or {})
                cap["npz"] = absp(cap.get("npz"))
                info = {k: sel[k] for k in ("variant", "arm", "cid")}
                info.update(json=sel["json"], video=sel["video"], npz=cap.get("npz"),
                            onset=(cap.get("onset") or {}).get("t_event"), fall_rest_step=cap.get("fall_rest_step"),
                            settle_sim_steps=cap.get("settle_sim_steps"), in_final100=cap.get("npz") in final_npz)
                if cap.get("npz") and os.path.exists(cap["npz"]):
                    q = np.load(cap["npz"])["expert_state"]
                    step = float(np.abs(np.diff(q[:, ARM], axis=0)).max()) if len(q) > 1 else 0.0
                    oor = ((q < lo) | (q > hi))[:, ARM]
                    info.update(frames=int(len(q)), max_joint_step=round(step, 4), G7_pass=step <= G7_MAX,
                                oor_frames=int(oor.any(1).sum()), G8_pass=not oor.any(),
                                oor_joints=[ARM[i] for i in np.where(oor.any(0))[0]],
                                joint_min=np.round(q[:, ARM].min(0), 3).tolist(),
                                joint_max=np.round(q[:, ARM].max(0), 3).tolist())
                tag = f"{seed}_step{j['step']}"
                info["asset_files"] = {"json": keep(sel["json"], f"recoveries/{tag}.json"),
                                       "video": keep(sel["video"], f"recoveries/{tag}.mp4"),
                                       "npz": keep(cap.get("npz"), f"recoveries/{tag}.npz")}
                mr6["selected"] = info
            elif s["n_candidates"] > 0:
                tier = "B_feasible_unrecovered"
            else:
                tier = "C_no_feasible_chain"
        row["mr6_tier"], row["mr6"] = tier, mr6
        ep = f"{EXPERT}/mugS_{seed}.json"
        row["expert_demo"] = None
        if os.path.exists(ep):
            ej = json.load(open(ep))
            row["expert_demo"] = {"json": ep, "npz": ep[:-5] + ".npz", "status": ej.get("status"), "arm": ej.get("arm"),
                                  "frames": ej.get("expert_frames"),
                                  "asset_files": {"json": keep(ep, f"expert/{seed}.json"),
                                                  "npz": keep(ep[:-5] + ".npz", f"expert/{seed}.npz")}}
        rows.append(row)
    with open(f"{out}/index.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tiers = {}
    for row in rows:
        tiers[row["mr6_tier"]] = tiers.get(row["mr6_tier"], 0) + 1
    A = [r for r in rows if r["mr6_tier"] == "A_recovered"]
    meta = {"name": "rf_hanging_mug_v1", "built": time.strftime("%F %T"), "n_states": len(rows), "tiers": tiers,
            "A_G8_pass": sum(bool(r["mr6"]["selected"].get("G8_pass")) for r in A),
            "A_G7_pass": sum(bool(r["mr6"]["selected"].get("G7_pass")) for r in A),
            "A_in_final100": sum(bool(r["mr6"]["selected"].get("in_final100")) for r in A),
            "expert_demo_success": sum(bool(r["expert_demo"] and r["expert_demo"]["status"] == "SUCCESS") for r in rows),
            "G8_range": {"lo": np.round(lo, 3).tolist(), "hi": np.round(hi, 3).tolist(), "margin": MARGIN,
                         "source": "frozen-list expert demos + FACT failure prefixes"},
            "provenance": {"jobs": JOBS, "selections": SEL, "planner": "MR6 (code/mug_recovery_chain.py)",
                           "takeover_rule": "code/mug_takeover_step.py", "curobo": "0.7.8",
                           "code_sha256": {f: sha(f"{L}/code/{f}") for f in
                                           ("takeover_bench.py", "mug_recovery_chain.py", "mug_takeover_step.py",
                                            "takeover_fixes.py", "sim_select.py", "quality_gates.py")},
                           "hanging_mug_env_sha256": sha(f"{L}/RoboTwin-FACT/envs/hanging_mug.py")}}
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("n_states", "tiers", "A_G8_pass", "A_G7_pass", "A_in_final100",
                                           "expert_demo_success")}))


if __name__ == "__main__":
    main(sys.argv[1])
