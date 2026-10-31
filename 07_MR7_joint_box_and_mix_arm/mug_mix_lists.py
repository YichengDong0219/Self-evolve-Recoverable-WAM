"""Build the npz_to_lerobot input lists for the hanging_mug Mix arm (50 recovery + 50 expert) (Claude, 2026-10-30).

usage: python mug_mix_lists.py <out dir> [path_prefix_from=path_prefix_to]
Seeds = the 100 seeds of the existing E arm (sft_lists_final_g7_1029/paired.json), so both arms see the same scenes.
Valid recovery for a seed = a selected simulate-and-select recovery from any of the sources below that passes the
mug_sft_lists rules (capture npz exists, fall detected by a FALL rule, G7 on recovery and expert <= 0.2 rad, expert demo
SUCCESS) plus G8 (all arm joints inside the asset's G8 range, expert demos + FACT prefixes +-0.5 rad; the old R arm
collapsed on out-of-range recoveries). If a seed has several, the first source in SOURCES wins.
Rule fixed before the draw (10-30): 50 recovery seeds = random.Random(20261030).sample(sorted valid seeds, 50); the
other 50 E-arm seeds use their expert demo.
Writes <out dir>/{R_recovery.jsonl, R_prefix.jsonl, E_expert.jsonl, mix.json}; same row formats as mug_sft_lists.py.
"""
import glob
import json
import os
import random
import sys

import numpy as np

from mug_sft_lists import FALL_RULES, G7_MAX_STEP, _max_arm_step

L = "/workspace/fact_local"
SOURCES = [("MR6", f"{L}/bench/SEL_v7"), ("MR7_smoke", f"{L}/bench/SEL_mr7_smoke"),
           ("MR7_smoke_r2", f"{L}/bench/SEL_mr7_smoke_r2"), ("MR7_topup", f"{L}/bench/SEL_mr7_topup")]
E_LIST = f"{L}/sft_lists_final_g7_1029"
EXPERT = f"{L}/expert_mug"
DRAW_SEED, N_REC = 20261030, 50
ARM = [i for i in range(14) if i not in (6, 13)]


def absp(p):
    """cluster results carry /home paths (copied into the same tree here); MR6 paths may be relative to code/"""
    if p.startswith("/workspace_cluster/fact_local"):
        return p.replace("/workspace_cluster/fact_local", L)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(L, "code", p))


def check(sel_path, seed, g8lo, g8hi):
    """(row, None) for a valid recovery, (None, reason) otherwise"""
    r = json.load(open(sel_path))
    if not r.get("selected"):
        return None, "none"
    sel = r["selected"]
    cap = dict(json.load(open(absp(sel["json"]))).get("capture") or {})
    if not cap.get("npz") or not os.path.exists(absp(cap["npz"])):
        return None, "no_npz"
    cap["npz"] = absp(cap["npz"])
    ej = json.load(open(f"{EXPERT}/mugS_{seed}.json"))
    if ej["status"] != "SUCCESS":
        return None, "expert_fail"
    ar = (cap.get("onset") or {}).get("all_rules", {})
    falls = [ar[k] for k in FALL_RULES if k in ar]
    if not falls:
        return None, "no_fall"
    t_on = min(falls)
    if t_on != cap["onset"]["t_event"]:  # same recomputation as mug_sft_lists
        P = np.load(cap["npz"])["prefix_mug_p"]
        cap["fall_rest_step"] = next((t for t in range(t_on, len(P) - 10) if np.linalg.norm(P[t + 10] - P[t]) < 1e-3),
                                     len(P) - 1)
    q = np.load(cap["npz"])["expert_state"]
    jr = max(_max_arm_step(q), _max_arm_step(np.load(f"{EXPERT}/mugS_{seed}.npz")["expert_state"]))
    if jr > G7_MAX_STEP:
        return None, f"G7 {jr:.3f}"
    if ((q < g8lo) | (q > g8hi))[:, ARM].any():
        return None, "G8"
    moving = np.abs(np.diff(q, axis=0)).max(axis=1) > 1e-4
    return {"npz": cap["npz"], "instruction": cap["instruction"], "env_seed": seed,
            "action_masked_until": int(np.argmax(moving)) if moving.any() else len(q),
            "end": cap["fall_rest_step"], "failure_active_from": t_on, "arm": sel["arm"],
            "flip": bool(sel.get("flip", False)), "variant": sel["variant"], "video": absp(sel["video"]),
            "selection": sel_path, "max_arm_step": round(jr, 4), "frames": int(len(q))}, None


def main():
    out = sys.argv[1]
    src, dst = (sys.argv[2].split("=") if len(sys.argv) > 2 else ("", ""))
    fix = (lambda p: dst + p[len(src):] if src and p.startswith(src) else p)
    meta = json.load(open(f"{L}/assets/rf_hanging_mug_v1/meta.json"))
    g8lo, g8hi = np.array(meta["G8_range"]["lo"]), np.array(meta["G8_range"]["hi"])
    seeds = [int(n.split("_")[1]) for n in json.load(open(f"{E_LIST}/paired.json"))["names"]]
    assert len(seeds) == len(set(seeds)) == 100
    pool, reasons = {}, {}
    for seed in seeds:
        for name, d in SOURCES:
            for p in sorted(glob.glob(f"{d}/*_{seed}/selection.json")):
                row, why = check(p, seed, g8lo, g8hi)
                if row:
                    pool.setdefault(seed, {**row, "source": name})
                else:
                    reasons.setdefault(seed, []).append(f"{name}:{why}")
            if seed in pool:
                break
    valid = sorted(pool)
    assert len(valid) >= N_REC, f"only {len(valid)} valid recoveries"
    rec = sorted(random.Random(DRAW_SEED).sample(valid, N_REC))
    exp = [s for s in seeds if s not in rec]
    os.makedirs(out, exist_ok=False)
    with open(f"{out}/R_recovery.jsonl", "w") as fr, open(f"{out}/R_prefix.jsonl", "w") as fp:
        for s in rec:
            x = pool[s]
            fr.write(json.dumps({"npz": fix(x["npz"]), "instruction": x["instruction"], "env_seed": s,
                                 "action_masked_until": x["action_masked_until"]}) + "\n")
            fp.write(json.dumps({"npz": fix(x["npz"]), "instruction": x["instruction"], "env_seed": s,
                                 "end": x["end"], "failure_active_from": x["failure_active_from"]}) + "\n")
    e_rows = {json.loads(x)["env_seed"]: json.loads(x) for x in open(f"{E_LIST}/E_expert.jsonl")}
    with open(f"{out}/E_expert.jsonl", "w") as fe:
        for s in exp:
            fe.write(json.dumps({**e_rows[s], "npz": fix(e_rows[s]["npz"])}) + "\n")
    json.dump({"rule": f"random.Random({DRAW_SEED}).sample(sorted valid seeds, {N_REC})", "e_arm_seeds": seeds,
               "valid": valid, "recovery_seeds": rec, "expert_seeds": exp,
               "pool": {str(s): pool[s] for s in valid}, "invalid_reasons": {str(k): v for k, v in reasons.items()
                                                                              if k not in pool}},
              open(f"{out}/mix.json", "w"), indent=1)
    by = lambda k: {v: sum(pool[s][k] == v for s in rec) for v in sorted({pool[s][k] for s in rec}, key=str)}
    print(f"valid {len(valid)}/100; drawn {len(rec)} recovery + {len(exp)} expert; "
          f"source {by('source')}; arm {by('arm')}; flip {by('flip')}")


if __name__ == "__main__":
    main()
