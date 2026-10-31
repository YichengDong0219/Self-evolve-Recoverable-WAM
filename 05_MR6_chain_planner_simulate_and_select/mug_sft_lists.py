"""Build the npz_to_lerobot input lists for the hanging_mug E vs R comparison (Claude, 2026-10-29).

usage: python mug_sft_lists.py <SEL dir> <expert dir> <out dir> [n=100] [path_prefix_from=path_prefix_to]
Paired set = selected recovery states whose capture npz exists AND whose expert demo on the same scene succeeded;
the first n by selection time (selection.json mtime) are used, so neither arm chooses its own episodes.
Writes <out dir>/{R_recovery.jsonl, R_prefix.jsonl, E_expert.jsonl, paired.json}:
  R_recovery  key "expert" = the curobo recovery segment (settle frames kept); action_masked_until = curobo's first
              motion frame: frames before it train the world model only (action_mask_patch.py, user's choice 10-29)
  R_prefix    key "prefix" = the FACT rollout up to the end of the fall (mug at rest), FACT failure episode with
              failure_active_from = fall onset (value penalized during the fall only; no action loss on the prefix)
Fall onset = earliest FALL rule of mug_failure_onset (a decoupled fall / b back to table). The capture's own onset is
the earliest rule of any kind, so a "c stall" before a later drop would end the prefix before the fall (2 cases,
10-29): those get the fall onset and the rest step recomputed with takeover_bench's definition (first t >= onset with
|p[t+10] - p[t]| < 1 mm). States with no detected fall (onset None or stall only) are not used: the agreed value
penalty ("only the falling process") is undefined for them. G7: pairs with a single-frame arm step > 0.2 rad are not used.
  E_expert    key "expert" = stock expert demo from the initial scene
The optional path rewrite maps npz paths to their location on the training machine.
"""
import glob
import json
import os
import sys

import numpy as np

FALL_RULES = ("a_decoupled_fall", "b_back_to_table")
G7_MAX_STEP = 0.2
ARM = [i for i in range(14) if i not in (6, 13)]


def _max_arm_step(state):
    return float(np.abs(np.diff(state, axis=0))[:, ARM].max())


def main():
    sel_dir, exp_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    src, dst = (sys.argv[5].split("=") if len(sys.argv) > 5 else ("", ""))
    fix = (lambda p: dst + p[len(src):] if src and p.startswith(src) else p)
    rows, skipped = [], []
    here = os.path.dirname(os.path.abspath(__file__))
    absp = (lambda q: q if os.path.isabs(q) else os.path.normpath(os.path.join(here, q)))  # sim_select run from code/
    for p in glob.glob(os.path.join(sel_dir, "*", "selection.json")):
        r = json.load(open(p))
        if not r.get("selected"):
            continue
        cap = dict(json.load(open(absp(r["selected"]["json"]))).get("capture") or {})
        if cap.get("npz"):
            cap["npz"] = absp(cap["npz"])
        e = os.path.join(exp_dir, r["name"] + ".json")
        if not (cap.get("npz") and os.path.exists(cap["npz"]) and os.path.exists(e)):
            continue
        ej = json.load(open(e))
        if ej["status"] != "SUCCESS":
            continue
        ar = (cap.get("onset") or {}).get("all_rules", {})
        falls = [ar[k] for k in FALL_RULES if k in ar]
        if not falls:
            skipped.append((r["name"], "no_fall:" + str((cap.get("onset") or {}).get("rule"))))
            continue
        t_on = min(falls)
        if t_on != cap["onset"]["t_event"]:
            P = np.load(cap["npz"])["prefix_mug_p"]
            cap["fall_rest_step"] = next((t for t in range(t_on, len(P) - 10) if np.linalg.norm(P[t + 10] - P[t]) < 1e-3),
                                         len(P) - 1)
            skipped.append((r["name"], f"onset {cap['onset']['t_event']} -> fall {t_on}, rest {cap['fall_rest_step']}"))
        cap["fall_onset"] = t_on
        # G7 (user 10-29): no single-frame arm-joint step > 0.2 rad (expert data max 0.13). Recovery segments with an
        # IK-configuration snap at a plan start (0.24-0.52 rad in one frame) are not used.
        jr = max(_max_arm_step(np.load(cap["npz"])["expert_state"]),
                 _max_arm_step(np.load(os.path.join(exp_dir, r["name"] + ".npz"))["expert_state"]))
        if jr > G7_MAX_STEP:
            skipped.append((r["name"], f"G7 arm step {jr:.3f} rad"))
            continue
        rows.append((os.path.getmtime(p), r["name"], cap, ej))
    rows.sort()
    for x in skipped:
        print("note", *x)
    print("paired available", len(rows), "using", min(n, len(rows)))
    rows = rows[:n]
    os.makedirs(out, exist_ok=False)
    with open(f"{out}/R_recovery.jsonl", "w") as fr, open(f"{out}/R_prefix.jsonl", "w") as fp, \
            open(f"{out}/E_expert.jsonl", "w") as fe:
        for _, name, cap, ej in rows:
            seed = ej["env_seed"]
            st = np.load(cap["npz"])["expert_state"]
            moving = np.abs(np.diff(st, axis=0)).max(axis=1) > 1e-4
            m = int(np.argmax(moving)) if moving.any() else len(st)  # action[m] = state[m+1] is curobo's first motion
            fr.write(json.dumps({"npz": fix(cap["npz"]), "instruction": cap["instruction"], "env_seed": seed,
                                 "action_masked_until": m}) + "\n")
            fp.write(json.dumps({"npz": fix(cap["npz"]), "instruction": cap["instruction"], "env_seed": seed,
                                 "end": cap["fall_rest_step"], "failure_active_from": cap["fall_onset"]}) + "\n")
            assert ej["instruction"] == cap["instruction"], name  # same scene, same instruction text in both arms
            fe.write(json.dumps({"npz": fix(os.path.abspath(os.path.join(exp_dir, name + ".npz"))),
                                 "instruction": ej["instruction"], "env_seed": seed}) + "\n")
    json.dump({"names": [r[1] for r in rows], "notes": skipped}, open(f"{out}/paired.json", "w"), indent=1)


if __name__ == "__main__":
    main()
