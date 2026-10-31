"""Where does the 0.1-0.2 rad pre-grasp orientation error come from? (Claude, 2026-10-28). No model.

Replays a hanging_mug failure to --step, settles, then for the arm/contact set MR5 would choose builds grasp_actor's
actions and executes ONLY the move to the pre-grasp pose, from the current arm state (MR5) and, in a second process
run with --home, after back_to_origin (v4). Reports target vs reached end-effector pose (position error, rotation
error), the planned trajectory's last joint vector vs the joints actually reached, and the FK of the planned last
joints vs the target (planner tolerance).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import transforms3d as t3d

import g0_driver as G
import mug_recovery as MR
import mug_recovery_direct as MRD
import takeover_fixes as TF
from envs.utils import *  # noqa: F401,F403


def rot_err(q1, q2):
    d = abs(float(np.dot(np.asarray(q1) / np.linalg.norm(q1), np.asarray(q2) / np.linalg.norm(q2))))
    return float(2 * np.arccos(min(1.0, d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", type=Path, required=True)
    ap.add_argument("--step", type=int, default=336)
    ap.add_argument("--home", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    src = json.loads(a.rollout.read_text())
    args = G.build_args("hanging_mug", src["config"])
    TF.apply_before_env("seeds+best+rack")
    env = G.task_env("hanging_mug")
    seed, info = G.expert_seed(env, args, src["env_seed"], 1)
    G.setup_episode(env, args, seed, info, forced=src["instruction"])
    for act in src["executed_actions"][:a.step]:
        env.get_obs()
        env.take_action(np.asarray(act, dtype=np.float32))
    env.plan_success = True
    TF.apply_to_env(env, "seeds+best+rack")  # no pp: we want to see the raw error
    log = []
    MR.mug_rest(env, log)
    if a.home:
        env.move(env.back_to_origin(ArmTag("left")), env.back_to_origin(ArmTag("right")))  # noqa: F405
        MR.mug_rest(env, log)
    arm, cids = MRD.select(env, log)
    out = {"home": a.home, "arm": arm, "cids": cids}
    if arm is not None:
        tag = ArmTag(arm)  # noqa: F405
        planner = env.robot.left_planner if arm == "left" else env.robot.right_planner
        captured = {}
        orig = planner.plan_path

        def cap(*x, **k):
            r = orig(*x, **k)
            captured["res"] = r
            return r
        planner.plan_path = cap
        env.plan_success = True
        _, acts = env.grasp_actor(env.mug, arm_tag=tag, pre_grasp_dis=0.05, contact_point_id=cids)
        target = np.asarray(acts[0].target_pose, dtype=float)
        env._world_set(rack=True, mug=True)
        env.move((tag, acts[:1]))
        reached = np.asarray(env.robot.get_left_ee_pose() if arm == "left" else env.robot.get_right_ee_pose(), float)
        q_reached = np.asarray(env.robot.get_left_arm_jointState() if arm == "left" else env.robot.get_right_arm_jointState())[:6]
        res = captured.get("res", {})
        q_plan_last = np.asarray(res["position"][-1], float)[:6] if res.get("status") == "Success" else None
        out.update({"target": target.round(4).tolist(), "reached": reached.round(4).tolist(),
                    "pos_err_m": round(float(np.linalg.norm(reached[:3] - target[:3])), 4),
                    "rot_err_rad": round(rot_err(reached[3:], target[3:]), 4),
                    "plan_status": res.get("status"),
                    "joint_err_plan_vs_reached": None if q_plan_last is None else
                    round(float(np.abs(q_plan_last - q_reached).max()), 4)})
    env.close_env()
    a.out.write_text(json.dumps(out))
    print(json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
