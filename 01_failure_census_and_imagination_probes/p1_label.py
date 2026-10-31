"""P1 dense labelling replay (Claude, 2026-10-27). No model: replays a rollout's executed actions bit-exactly
(validated in G0) and records per-step task facts used to locate failure events offline.
Writes one new JSON per episode; refuses overwrite.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import g0_driver as G

TARGET = {"hanging_mug": "mug", "move_stapler_pad": "stapler", "turn_switch": "switch"}


def robot_links(env):
    names = set()
    for a in env.scene.get_all_articulations():
        if a.get_name() == "":  # the dual-arm robot articulation has an empty name in these scenes
            names |= {l.get_name() for l in a.get_links()}
    return names


def target_contacts(env, target_name, links):
    hit = set()
    for c in env.scene.get_contacts():
        n0, n1 = c.bodies[0].entity.name, c.bodies[1].entity.name
        if n0 == target_name and n1 in links:
            hit.add(n1)
        elif n1 == target_name and n0 in links:
            hit.add(n0)
    return sorted(hit)


def facts(env, task, links):
    r = env.robot
    row = {"step": int(env.take_action_cnt), "success": bool(env.check_success()),
           "grip": [round(float(r.left_gripper_val), 4), round(float(r.right_gripper_val), 4)],
           "tcp_l": np.round(r.get_left_tcp_pose()[:3], 4).tolist(),
           "tcp_r": np.round(r.get_right_tcp_pose()[:3], 4).tolist()}
    obj = getattr(env, TARGET[task])
    name = obj.get_name()
    row["contacts"] = target_contacts(env, name, links)
    if task == "turn_switch":
        lim = env.switch.get_qlimits()[0]
        row["q"] = round(float(env.switch.get_qpos()[0]), 5)
        row["q_lim"] = [round(float(lim[0]), 5), round(float(lim[1]), 5)]
    else:
        p = obj.get_pose()
        row["p"] = np.round(p.p, 4).tolist()
        row["quat"] = np.round(p.q, 4).tolist()
        zaxis = p.to_transformation_matrix()[:3, 2]
        row["tilt_deg"] = round(float(np.degrees(np.arccos(np.clip(abs(zaxis[2]), 0, 1)))), 2)
        if task == "hanging_mug":
            fp = env.mug.get_functional_point(0)[:3]
            rack_mid = (env.rack.get_pose().p + env.rack.get_functional_point(0)[:3]) / 2
            row["fp"] = np.round(fp, 4).tolist()
            row["dxy"] = round(float(np.linalg.norm((fp - rack_mid)[:2])), 4)
            row["dz"] = round(float(fp[2] - 0.86), 4)
        else:
            tgt = env.pad.get_pose().p
            row["dxy"] = round(float(np.linalg.norm((p.p - tgt)[:2])), 4)
            row["dz"] = round(float(p.p[2] - tgt[2]), 4)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    src = json.loads(a.rollout.read_text())
    task, cfg = src["task"], src["config"]
    args = G.build_args(task, cfg)
    env = G.task_env(task)
    # The expert check is not perfectly reproducible (seen on hanging_mug 100019); retry the SAME seed only.
    expert_attempts = 0
    for expert_attempts in range(1, 4):
        try:
            seed, info = G.expert_seed(env, args, src["env_seed"], 1)
            break
        except RuntimeError:
            if expert_attempts == 3:
                raise
    assert seed == src["env_seed"]
    G.setup_episode(env, args, seed, info, forced=src["instruction"])
    links = robot_links(env)
    rows = [facts(env, task, links)]
    for i, act in enumerate(src["executed_actions"]):
        env.get_obs()
        env.take_action(np.asarray(act, dtype=np.float32))
        rows.append(facts(env, task, links))
        if env.eval_success:
            break
    final = G.capture(env)
    ok = (bool(env.eval_success) == src["success"] and int(env.take_action_cnt) == src["steps"])
    env.close_env()
    out = {"format": "fact-p1-label/1", "rollout": str(a.rollout), "task": task, "config": cfg, "env_seed": seed,
           "success": src["success"], "steps": src["steps"], "replay_outcome_matches": ok,
           "expert_check_attempts": expert_attempts,
           "robot_links": sorted(links), "rows": rows,
           "final_state_matches": final["actors"] == src["states"]["final"]["actors"]}
    tmp = a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.rename(a.out)
    print(json.dumps({"rollout": a.rollout.name, "rows": len(rows), "replay_outcome_matches": ok,
                      "final_state_matches": out["final_state_matches"]}))


if __name__ == "__main__":
    sys.exit(main())
