"""Why does a resting mug have no reachable rim contact point? (Claude, 2026-10-28). No model.

Replays a failed hanging_mug rollout to --step, settles the mug and parks both arms (as mug_recovery.run does), then
for every contact point x arm x pre-grasp distance reports: grasp pose exists?, approach direction (world), pre-grasp
cuRobo status, grasp cuRobo status. Raw cuRobo statuses are read through takeover_diag-style wrapping.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import transforms3d as t3d

import g0_driver as G
import mug_recovery as MR
import takeover_fixes as TF
from envs.utils import *  # noqa: F401,F403


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", type=Path, required=True)
    ap.add_argument("--step", type=int, default=336)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    src = json.loads(a.rollout.read_text())
    args = G.build_args("hanging_mug", src["config"])
    TF.apply_before_env("pp+seeds+best")
    env = G.task_env("hanging_mug")
    for attempt in range(1, 4):
        try:
            seed, info = G.expert_seed(env, args, src["env_seed"], 1)
            break
        except RuntimeError:
            if attempt == 3:
                raise
    G.setup_episode(env, args, seed, info, forced=src["instruction"])
    for act in src["executed_actions"][:a.step]:
        env.get_obs()
        env.take_action(np.asarray(act, dtype=np.float32))
    env.plan_success = True
    TF.apply_to_env(env, "pp+seeds+best")
    log = []
    MR.mug_rest(env, log)
    env.move(env.back_to_origin(ArmTag("left")), env.back_to_origin(ArmTag("right")))  # noqa: F405
    MR.mug_rest(env, log)
    state = MR.classify(env)
    statuses = {}
    for arm, pl in (("left", env.robot.left_planner), ("right", env.robot.right_planner)):
        orig = pl.motion_gen.plan_single

        def wrap(*x, _o=orig, _arm=arm, **k):
            r = _o(*x, **k)
            statuses.setdefault(_arm, []).append(str(getattr(r, "status", None)).split(".")[-1])
            return r
        pl.motion_gen.plan_single = wrap
    rows = []
    for cid, _ in env.mug.iter_contact_points():
        cp = np.asarray(env.mug.get_contact_point(cid, "list"))
        cm = env.mug.get_contact_point(cid, "matrix") @ np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        approach_raw = np.round(cm[:3, 0], 2).tolist()  # direction the gripper moves along to reach the point
        for arm in ("left", "right"):
            plan = env.robot.left_plan_path if arm == "left" else env.robot.right_plan_path
            for pre_dis in (0.05,):
                env.plan_success = True
                pre = env.get_grasp_pose(env.mug, ArmTag(arm), contact_point_id=cid, pre_dis=pre_dis)  # noqa: F405
                row = {"cid": cid, "arm": arm, "pre_dis": pre_dis, "cp": np.round(cp[:3], 3).tolist(), "approach_raw": approach_raw,
                       "pose": pre is not None}
                if pre is not None:
                    R = t3d.quaternions.quat2mat(pre[-4:])
                    row["approach"] = np.round(R[:, 0], 2).tolist()  # gripper x axis = approach direction
                    n0 = len(statuses.get(arm, []))
                    row["pre_ok"] = plan(list(pre))["status"] == "Success"
                    row["pre_status"] = (statuses.get(arm, [])[n0:] or ["?"])[-1]
                rows.append(row)
    env.close_env()
    out = {"rollout": str(a.rollout), "seed": seed, "state": state, "rows": rows}
    a.out.write_text(json.dumps(out, default=float))
    print(json.dumps(state, default=float))
    for r in rows:
        print(r)


if __name__ == "__main__":
    sys.exit(main())
