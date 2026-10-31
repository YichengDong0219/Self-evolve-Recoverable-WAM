"""Loader for the reusable recoverable-failure assets (Claude, 2026-10-30). See assets/rf_hanging_mug_v1/README.md.

  import rf_assets as RF
  rows = RF.load(tiers=["A_recovered", "B_feasible_unrecovered"])
  RF.to_jobs(rows, "bench/jobs_mr7_pilot.jsonl", group="mr7", plan="MR6")   # -> sim_select.py job list
CLI: python rf_assets.py <tiers comma-separated|all> <jobs_out.jsonl> <group> [plan=MR6] [limit]
"""
import hashlib
import json
import os
import sys

L = "/workspace/fact_local"
ASSET = f"{L}/assets/rf_hanging_mug_v1"
TIERS = ["A_recovered", "B_feasible_unrecovered", "C_no_feasible_chain", "U_untried"]


def load(asset=ASSET, tiers=None, verify=False):
    rows = [json.loads(x) for x in open(f"{asset}/index.jsonl")]
    rows = [r for r in rows if tiers is None or r["mr6_tier"] in tiers]
    for r in rows:
        r["rollout_path"] = os.path.join(asset, r["rollout"])
        if verify:  # the rollout must be byte-identical to the one the verdicts were obtained with
            h = hashlib.sha256(open(r["rollout_path"], "rb").read()).hexdigest()
            assert h == r["rollout_sha256"], r["state_id"]
    return rows


def to_jobs(rows, out, group, plan="MR6"):
    """sim_select.py job lines; use a new group name and a new out_dir per planner iteration."""
    assert not os.path.exists(out), f"{out} exists"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps({"rollout": r["rollout_path"], "step": r["takeover_step"], "plan": plan,
                                "group": group, "onset": r["failure_type"], "state_id": r["state_id"]}) + "\n")
    return len(rows)


if __name__ == "__main__":
    t = None if sys.argv[1] == "all" else sys.argv[1].split(",")
    rows = load(tiers=t, verify=True)
    if len(sys.argv) > 5:
        rows = rows[:int(sys.argv[5])]
    print(to_jobs(rows, sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "MR6"), "jobs")
