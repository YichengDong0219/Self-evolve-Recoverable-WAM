"""FACT G0 branching driver (Claude, 2026-10-27). Runs in the RoboTwin env, cwd = RoboTwin-FACT.

Modes (one episode per invocation, explicit env seed found by the official expert check):
  rollout  closed loop with the probe server; explicit continuation seeds per decision; records every executed
           qpos command and a physics/task state capture at every plan boundary (step % 48 == 0) and at the end.
  replay   no model: re-executes a recorded action list on a freshly set-up scene, capturing the same boundaries.
  branch   replays the first --branch-step actions, then hands over to the probe server with the ORIGINAL
           continuation seeds (decision index aligned), i.e. a branch that must reproduce the original suffix.
Output: one new JSON file (refuses to overwrite). State capture is raw (SAPIEN 3 API); comparison is offline.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")
import local.robotwin_compat  # noqa: F401,E402  FACT host CUDA/Warp compatibility (same as eval_policy.py)
import yaml  # noqa: E402
from envs import CONFIGS_PATH  # noqa: E402
from envs.utils.create_actor import UnStableError  # noqa: E402
from generate_episode_instructions import generate_episode_descriptions  # noqa: E402
from sapien import physx  # noqa: E402

PLAN = 48


def build_args(task_name, task_config):
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args.update(task_name=task_name, task_config=task_config, ckpt_setting="g0")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        cam = yaml.load(f.read(), Loader=yaml.FullLoader)
    head = args["camera"]["head_camera_type"]
    args["head_camera_h"], args["head_camera_w"] = cam[head]["h"], cam[head]["w"]
    et = args["embodiment"]
    if len(et) != 1:
        raise ValueError("G0 driver supports single-embodiment configs only")
    robot_file = emb[et[0]]["file_path"]
    args["left_robot_file"] = args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        ecfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["left_embodiment_config"] = args["right_embodiment_config"] = ecfg
    args["eval_video_log"] = False
    args["eval_mode"] = True
    args["render_freq"] = 0
    return args


def task_env(task_name):
    mod = __import__(f"envs.{task_name}", fromlist=[task_name])
    return getattr(mod, task_name)()


def expert_seed(env, args, start_seed, max_tries):
    """Official expert check (same as eval_policy.py): first seed whose expert plan succeeds."""
    for seed in range(start_seed, start_seed + max_tries):
        try:
            env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
            info = env.play_once()
            ok = env.plan_success and env.check_success()
            env.close_env()
        except UnStableError:
            env.close_env(); continue
        except Exception:  # noqa: BLE001 - mirrors eval_policy.py's skip-on-error behaviour
            env.close_env(); continue
        if ok:
            return seed, info
    raise RuntimeError(f"no expert-valid seed in [{start_seed}, {start_seed + max_tries})")


def setup_episode(env, args, seed, info, instruction_type="unseen", forced=None, override=None):
    env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    # Upstream draws instruction wording with the never-seeded Python `random` module, so the instruction
    # changes between processes. G0 protocol: seed it with the scene seed (wording fixed per scene).
    random.seed(seed)
    results = generate_episode_descriptions(args["task_name"], [info["info"]], 1)
    instruction = str(np.random.choice(results[0][instruction_type]))
    if override is not None:  # e.g. the exact text another policy was evaluated with on this scene
        instruction = str(override)
    if forced is not None and instruction != forced:
        raise RuntimeError(f"instruction differs from the source rollout: {instruction!r} != {forced!r}")
    env.set_instruction(instruction=instruction)
    return instruction


def _h(obj):
    return hashlib.sha256(repr(obj).encode()).hexdigest()


def capture(env):
    actors = {}
    for i, e in enumerate(env.scene.get_all_actors()):
        p = e.get_pose()
        row = {"p": p.p.tolist(), "q": p.q.tolist()}
        for c in e.get_components():
            if isinstance(c, physx.PhysxRigidDynamicComponent):
                row["v"] = np.asarray(c.get_linear_velocity()).tolist()
                row["w"] = np.asarray(c.get_angular_velocity()).tolist()
                row["kinematic"] = bool(c.get_kinematic())
        actors[f"{i}:{e.get_name()}"] = row
    arts = {}
    for i, a in enumerate(env.scene.get_all_articulations()):
        joints = a.get_active_joints()
        rp = a.get_root_pose()
        arts[f"{i}:{a.get_name()}"] = {
            "root_p": rp.p.tolist(), "root_q": rp.q.tolist(),
            "qpos": np.asarray(a.get_qpos()).tolist(), "qvel": np.asarray(a.get_qvel()).tolist(),
            "drive_target": [np.asarray(j.get_drive_target()).reshape(-1).tolist() for j in joints],
            "drive_velocity_target": [np.asarray(j.get_drive_velocity_target()).reshape(-1).tolist() for j in joints],
            "joint_types": [j.get_type() for j in joints],
        }
    r = env.robot
    return {
        "action_step": int(env.take_action_cnt),
        "eval_success": bool(env.eval_success),
        "check_success": bool(env.check_success()),
        "gripper_val": [float(r.left_gripper_val), float(r.right_gripper_val)],
        "np_rng_sha": _h(np.random.get_state()[1].tolist()) + ":" + str(np.random.get_state()[2]),
        "py_rng_sha": _h(random.getstate()),
        "actors": actors,
        "articulations": arts,
    }


def make_policy(port):
    sys.path.insert(0, os.environ["FACT_PROBE_DIR"])
    from evaluation.robotwin import model2robotwin_interface as mi
    from client_adapter import probe_client_class
    base = mi.get_model({"port": port, "execute_actions_per_plan": PLAN, "server_timeout_ms": 300000,
                         "server_wait_seconds": 600, "enable_sample": False, "best_of_n": 1})
    base.__class__ = probe_client_class(type(base))
    return mi, base


IMAGINED = []  # per decision: uint8 [4, H, W, 3] predicted keyframes (only when the server runs --return_images)


def server_sink(io_log, save_request=None):
    def sink(kind, msg):
        if kind == "response" and msg.get("images") is not None:
            im = msg["images"]
            im = im.detach().cpu().float().numpy() if hasattr(im, "detach") else np.asarray(im, dtype=np.float32)
            im = im[0]  # C, T(=48 expanded), H, W ; keyframe i is held on steps 12i..12i+11
            keys = [np.clip((np.transpose(im[:, t], (1, 2, 0)) + 1) / 2 * 255, 0, 255).astype(np.uint8)
                    for t in range(0, im.shape[1], 12)]
            IMAGINED.append(np.stack(keys))
        if kind == "request" and save_request is not None and not save_request.exists():
            arrays = {k.replace(".", "__"): np.asarray(v) for k, v in msg.items() if k.startswith("observation.")}
            np.savez(save_request, instruction=np.asarray(msg["instruction"]), **arrays)
        row = {"kind": kind, "probe_decision": msg.get("probe_decision"), "probe_seeds": msg.get("probe_seeds")}
        if kind == "response":
            act = msg["action"]
            act = act.detach().cpu().numpy() if hasattr(act, "detach") else np.asarray(act)
            row["action_sha256"] = hashlib.sha256(np.ascontiguousarray(act, dtype=np.float32).tobytes()).hexdigest()
            row["action"] = np.asarray(act, dtype=np.float32).reshape(-1, act.shape[-1]).tolist()
        io_log.append(row)
    return sink


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rollout", "replay", "branch"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--config", default="demo_clean")
    ap.add_argument("--start-seed", type=int, required=True)
    ap.add_argument("--max-tries", type=int, default=10)
    ap.add_argument("--root-seed", type=int, default=20261027)
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--actions-from", type=Path, help="rollout JSON whose executed actions are replayed")
    ap.add_argument("--branch-step", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-request", type=Path, help="save the first model request (observation) as .npz")
    ap.add_argument("--instruction", help="override the generated instruction with this exact text")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    args = build_args(a.task, a.config)
    env = task_env(a.task)
    seed, info = expert_seed(env, args, a.start_seed, a.max_tries)
    src = json.loads(a.actions_from.read_text()) if a.actions_from else None
    if a.mode != "rollout":
        if src is None or src["env_seed"] != seed or src["task"] != a.task or src["config"] != a.config:
            raise SystemExit("replay/branch needs a rollout JSON for the same task/config/env seed")
    actions = src["executed_actions"] if src else []
    if a.mode == "branch" and (a.branch_step % PLAN or not 0 < a.branch_step < len(actions)):
        raise SystemExit("branch step must be a plan boundary inside the recorded episode")
    stop_replay = {"replay": len(actions), "branch": a.branch_step, "rollout": 0}[a.mode]

    io_log = []
    start_id = f"{a.task}/{a.config}/{seed}"
    override = a.instruction if a.instruction is not None else (
        src["instruction"] if src and src.get("instruction_source") == "override" else None)
    instruction = setup_episode(env, args, seed, info, forced=src["instruction"] if src else None, override=override)
    mi = policy = None
    if a.mode != "replay":
        mi, policy = make_policy(a.port)
        mi.reset_model(policy)                       # reset clears the binding; bind afterwards
        policy.bind_probe(start_id, a.root_seed, server_sink(io_log, a.save_request))
        policy._probe_decision = stop_replay // PLAN  # align continuation seeds with the original decisions

    executed, states = [], {}
    while env.take_action_cnt < env.step_lim:
        step = int(env.take_action_cnt)
        if step % PLAN == 0:
            states[str(step)] = capture(env)
        obs = env.get_obs()
        if step < stop_replay:
            act = np.asarray(actions[step], dtype=np.float32)
            executed.append(act.tolist())
            env.take_action(act)
        elif policy is None:
            break
        else:
            before = len(executed)
            orig = env.take_action
            def rec(action, action_type="qpos", _orig=orig):  # noqa: E306
                executed.append(np.asarray(action, dtype=np.float32).tolist())
                return _orig(action, action_type=action_type)
            env.take_action = rec
            try:
                mi.eval(env, policy, obs)
            finally:
                del env.take_action  # drop the instance override, back to the (patched) class method
            if len(executed) != before + 1:
                raise RuntimeError("eval() did not execute exactly one action")
        if int(env.take_action_cnt) != step + 1:
            raise RuntimeError(f"step counter did not advance by one at step {step}")
        if env.eval_success:
            break
    states["final"] = capture(env)
    out = {"format": "fact-g0-episode/1", "mode": a.mode, "task": a.task, "config": a.config, "env_seed": seed,
           "instruction": instruction, "instruction_source": "override" if override is not None else "generated",
           "root_seed": a.root_seed, "start_id": start_id, "branch_step": stop_replay,
           "actions_from": str(a.actions_from) if a.actions_from else None,
           "success": bool(env.eval_success), "steps": int(env.take_action_cnt),
           "executed_actions": executed,
           "executed_actions_sha256": hashlib.sha256(json.dumps(executed).encode()).hexdigest(),
           "states": states, "server_io": io_log}
    env.close_env()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if IMAGINED:
        np.savez_compressed(a.out.with_suffix(".imagined.npz"), keyframes=np.stack(IMAGINED))
        out["imagined_decisions"] = len(IMAGINED)
    tmp = a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.rename(a.out)
    print(json.dumps({k: out[k] for k in ("mode", "task", "env_seed", "success", "steps", "branch_step",
                                          "executed_actions_sha256")}))


if __name__ == "__main__":
    main()
