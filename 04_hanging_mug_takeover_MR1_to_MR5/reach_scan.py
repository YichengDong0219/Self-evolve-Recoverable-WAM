"""Reachability scan around a resting mug (Claude, 2026-10-28). No model.

Replays a hanging_mug failure to --step, settles the mug, then asks cuRobo (table + rack world, nothing else) for a
plan from the current arm state to gripper poses on a grid around the mug: xy offsets from the mug center, fingertip
heights above the table, and the standard RoboTwin gripper orientations. Prints, per orientation and height, how many
grid points each arm reaches, and the raw cuRobo status of a few failures.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

import g0_driver as G
import mug_recovery as MR
import takeover_fixes as TF
from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from envs.utils import *  # noqa: F401,F403

ORIENTS = ["top_down", "top_down_little_left", "top_down_little_right", "left_arm_perf", "right_arm_perf", "front",
           "down_left", "down_right"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", type=Path, required=True)
    ap.add_argument("--step", type=int, default=336)
    a = ap.parse_args()
    src = json.loads(a.rollout.read_text())
    args = G.build_args("hanging_mug", src["config"])
    TF.apply_before_env("seeds+rack")
    env = G.task_env("hanging_mug")
    seed, info = G.expert_seed(env, args, src["env_seed"], 1)
    G.setup_episode(env, args, seed, info, forced=src["instruction"])
    for act in src["executed_actions"][:a.step]:
        env.get_obs()
        env.take_action(np.asarray(act, dtype=np.float32))
    env.plan_success = True
    TF.apply_to_env(env, "seeds+rack")
    MR.mug_rest(env, [])
    env._world_set(rack=True, mug=False)
    c = np.asarray(env.mug.get_pose().p)
    statuses = collections.defaultdict(list)
    for arm, pl in (("left", env.robot.left_planner), ("right", env.robot.right_planner)):
        orig = pl.motion_gen.plan_single

        def wrap(*x, _o=orig, _a=arm, **k):
            r = _o(*x, **k)
            statuses[_a].append(str(getattr(r, "status", None)).split(".")[-1])
            return r
        pl.motion_gen.plan_single = wrap
    print("mug center", np.round(c, 3).tolist(), "mouth dir", np.round(env.mug.get_pose().to_transformation_matrix()[:3, 1], 2).tolist())
    offs = [(dx, dy) for dx in (-0.1, -0.05, 0.0, 0.05, 0.1) for dy in (-0.1, -0.05, 0.0, 0.05, 0.1)]
    for o in ORIENTS:
        q = list(GRASP_DIRECTION_DIC[o])
        for h in (0.03, 0.08, 0.15):
            row = {}
            for arm in ("left", "right"):
                plan = env.robot.left_plan_path if arm == "left" else env.robot.right_plan_path
                n = 0
                st = collections.Counter()
                for dx, dy in offs:
                    env.plan_success = True
                    k0 = len(statuses[arm])
                    ok = plan([c[0] + dx, c[1] + dy, MR.TABLE_Z + h] + q)["status"] == "Success"
                    n += ok
                    if not ok and len(statuses[arm]) > k0:
                        st[statuses[arm][-1]] += 1
                row[arm] = (n, dict(st.most_common(2)))
            print(f"{o:22s} h={h:.2f}  left {row['left']}  right {row['right']}", flush=True)
    env.close_env()


if __name__ == "__main__":
    sys.exit(main())
