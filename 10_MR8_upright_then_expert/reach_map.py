"""Table-wide reachability map for MR8 design (Claude, 2026-11-01). No model, no replay.

usage: python reach_map.py <out.json> [variant=seeds+jboxz]
From the initial scene of one hanging_mug seed (arms at home), asks cuRobo (table only, rack/mug obstacles off) for a
plan to fingertip positions on a table grid, for gripper orientations between top-down (pitch 0) and horizontal
pointing away from the robot (pitch 90), at two fingertip heights. Result: per arm, per (pitch, height), the grid of
reachable cells, to see how far back on the table each kind of grasp or push can act.
"""
import json
import sys

import numpy as np
import transforms3d as t3d

import g0_driver as G
import takeover_fixes as TF
from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC

XS = np.round(np.arange(-0.35, 0.351, 0.05), 3)
YS = np.round(np.arange(-0.20, 0.351, 0.05), 3)
PITCHES = (0, 30, 45, 60, 90)
HEIGHTS = (0.04, 0.10)
TABLE_Z = 0.74


def orient(pitch):
    """Slerp between RoboTwin's top_down and front gripper quaternions (approach = gripper local x)."""
    q0, q1 = np.asarray(GRASP_DIRECTION_DIC["top_down"], float), np.asarray(GRASP_DIRECTION_DIC["front"], float)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    t = pitch / 90.0
    om = np.arccos(np.clip(np.dot(q0, q1), -1, 1))
    q = q0 if om < 1e-6 else (np.sin((1 - t) * om) * q0 + np.sin(t * om) * q1) / np.sin(om)
    return q / np.linalg.norm(q)


def main(out, variant="seeds+jboxz"):
    parts = variant.split("+")
    if "jboxz" in parts:
        import pathlib
        TF.BOX = json.load(open(pathlib.Path(__file__).with_name("mr8_joint_box_z55.json")))
        variant = "+".join("jbox" if p == "jboxz" else p for p in parts)
    TF.apply_before_env(variant)
    env = G.task_env("hanging_mug")
    args = G.build_args("hanging_mug", "demo_clean")
    seed, info = G.expert_seed(env, args, 100001, 5)
    G.setup_episode(env, args, seed, info)
    res = {"variant": variant, "seed": seed, "xs": XS.tolist(), "ys": YS.tolist(), "maps": {}}
    for arm in ("left", "right"):
        plan = env.robot.left_plan_path if arm == "left" else env.robot.right_plan_path
        for pitch in PITCHES:
            q = orient(pitch)
            a = t3d.quaternions.quat2mat(q)[:, 0]
            for h in HEIGHTS:
                grid = []
                for y in YS:
                    row = []
                    for x in XS:
                        tip = np.array([x, y, TABLE_Z + h])
                        ee = tip - 0.12 * a
                        env.plan_success = True
                        row.append(int(plan(list(ee) + list(q))["status"] == "Success"))
                    grid.append(row)
                res["maps"][f"{arm}|p{pitch}|h{h}"] = grid
                print(arm, pitch, h, "reachable", int(np.sum(grid)), flush=True)
    env.close_env()
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:])
