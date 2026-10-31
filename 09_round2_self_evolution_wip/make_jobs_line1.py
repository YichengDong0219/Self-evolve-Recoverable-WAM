"""Job list for line 1: more base-policy cuRobo recoveries (Claude, 2026-10-31, user request).

usage: python make_jobs_line1.py <out jobs.jsonl>
Rule fixed before any run: states of rf_hanging_mug_v1 with MR6 tier A or B that are not among the Mix recovery
seeds and were never tried by MR7 (SEL_mr7_smoke, SEL_mr7_smoke_r2, SEL_mr7_topup), and whose failure is a hang
failure (the mug fell after the first right-gripper release; handover failures are excluded by the user's choice).
Tier A first, then B, each by state_id. Same job format as jobs_mr7_topup.jsonl.
"""
import glob
import json
import os
import sys

L = "/workspace/fact_local"
A = f"{L}/assets/rf_hanging_mug_v1"
MR7_DIRS = ["SEL_mr7_smoke", "SEL_mr7_smoke_r2", "SEL_mr7_topup"]


def grip_events(acts):
    ev = []
    for g, n in ((6, "L"), (13, "R")):
        st = acts[0][g] > 0.5
        for k in range(1, len(acts)):
            v = acts[k][g]
            ns = True if v > 0.5 else (False if v < 0.3 else st)
            if ns != st:
                ev.append((k, n + ("o" if ns else "c")))
                st = ns
    return sorted(ev)


def phase(ro):
    """hang (mug lifted by the right arm, fell after the first right release) / handover / other"""
    ev = grip_events(ro["executed_actions"])
    zs = sorted((int(k), v["actors"]["0:039_mug"]["p"][2]) for k, v in ro["states"].items() if k != "final")
    z0 = zs[0][1]
    rc = next((k for k, e in ev if e == "Rc"), None)
    rel = next((k for k, e in ev if e == "Ro"), None)
    if rc is None:
        return "other"
    after = [z for t, z in zs if t >= rc + 48]
    if not after or max(after) < z0 + 0.05:
        return "handover"
    peak_t = max((z, t) for t, z in zs if t >= rc)[1]
    fall_t = next((t for t, z in zs if t > peak_t and z < z0 + 0.05), None)
    return "hang" if fall_t is not None and rel is not None and fall_t >= rel else "other"


def main(out):
    mix = json.load(open(f"{L}/sft_lists_mix_1030b/mix.json"))
    used = set(mix["recovery_seeds"])
    tried = {int(os.path.basename(os.path.dirname(p)).split("_")[-1])
             for d in MR7_DIRS for p in glob.glob(f"{L}/bench/{d}/*/selection.json")}
    jobs, skip = [], {}
    for r in sorted((json.loads(x) for x in open(f"{A}/index.jsonl")), key=lambda r: (r["mr6_tier"], r["state_id"])):
        why = None
        if r["mr6_tier"][0] not in "AB":
            why = "tier"
        elif r["env_seed"] in used:
            why = "in_mix"
        elif r["env_seed"] in tried:
            why = "mr7_tried"
        else:
            p = r["rollout"] if r["rollout"].startswith("/") else f"{A}/rollouts/" + os.path.basename(r["rollout"])
            if phase(json.load(open(p))) != "hang":
                why = "not_hang"
        if why:
            skip[why] = skip.get(why, 0) + 1
            continue
        jobs.append({"rollout": f"{A}/rollouts/" + os.path.basename(r["rollout"]), "step": r["takeover_step"],
                     "plan": "MR6", "group": "l1", "onset": r["failure_type"], "state_id": r["state_id"],
                     "tier": r["mr6_tier"]})
    assert not os.path.exists(out), f"refusing to overwrite {out}"
    with open(out, "w") as f:
        f.write("".join(json.dumps(j) + "\n" for j in jobs))
    print(f"{len(jobs)} jobs (A {sum(j['tier'][0] == 'A' for j in jobs)}, B {sum(j['tier'][0] == 'B' for j in jobs)}); "
          f"skipped {skip}")


if __name__ == "__main__":
    main(sys.argv[1])
