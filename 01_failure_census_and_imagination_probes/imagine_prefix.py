"""Reconstruct FACT's imagination for the policy part (steps 0..k) of a failure that the expert later recovered
(Claude, 2026-10-27).

Same method as imagine_episode.py (which stays unchanged): replay the recorded rollout bit-exactly and, at every
decision boundary before the takeover step k, send the identical request (three cameras, joint state, instruction,
ORIGINAL explicit seed) to a server started with --return_images. Stops at k instead of running the whole episode.
Outputs (stem --out):
  .real.npz      real head / left / right camera frames for steps 0..k (uint8, native resolution)
  .imagined.npz  per decision the 4 distinct imagined keyframes (D, 4, H, W, 3)
  .json          per-decision step, seed, regenerated-vs-executed action difference, value
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import g0_driver as G

PLAN = 48


def to_u8(img_chw_m11):
    x = np.transpose(np.asarray(img_chw_m11, dtype=np.float32), (1, 2, 0))
    return np.clip((x + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", type=Path, required=True)
    ap.add_argument("--stop-step", type=int, required=True, help="takeover step k; frames 0..k are kept")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--out", type=Path, required=True, help="output stem")
    a = ap.parse_args()
    if a.out.with_suffix(".json").exists():
        raise SystemExit("refusing to overwrite")
    from evaluation.robotwin.model2robotwin_interface import _to_chw01
    from world_action_model.sockets import RobotInferenceClient

    src = json.loads(a.rollout.read_text())
    seeds = [r["probe_seeds"] for r in src["server_io"] if r["kind"] == "response"]
    task = src["task"]
    args = G.build_args(task, src["config"])
    env = G.task_env(task)
    for attempt in range(1, 4):  # expert check is not perfectly reproducible; retry the SAME seed only
        try:
            seed, info = G.expert_seed(env, args, src["env_seed"], 1)
            break
        except RuntimeError:
            if attempt == 3:
                raise
    G.setup_episode(env, args, seed, info, forced=src["instruction"],
                    override=src["instruction"] if src.get("instruction_source") == "override" else None)
    client = RobotInferenceClient(host="127.0.0.1", port=a.port, timeout_ms=600000)
    assert client.ping()

    actions = src["executed_actions"]
    real, imagined, decisions = [], [], []
    for step in range(a.stop_step + 1):
        obs = env.get_obs()
        cams = [obs["observation"][k]["rgb"][..., :3] for k in ("head_camera", "left_camera", "right_camera")]
        real.append(np.stack([np.asarray(c, dtype=np.uint8) for c in cams]))
        if step == a.stop_step:
            break
        if step % PLAN == 0:
            d = step // PLAN
            req = {"observation.state": np.asarray(obs["joint_action"]["vector"], dtype=np.float32).reshape(-1),
                   "observation.images.cam_high": _to_chw01(obs["observation"]["head_camera"]["rgb"]),
                   "observation.images.cam_left_wrist": _to_chw01(obs["observation"]["left_camera"]["rgb"]),
                   "observation.images.cam_right_wrist": _to_chw01(obs["observation"]["right_camera"]["rgb"]),
                   "instruction": src["instruction"], "best_of_N": 1, "probe_seeds": list(seeds[d])}
            resp = client.inference(req)
            act = resp["action"]
            act = np.asarray(act.detach().cpu().float().numpy() if hasattr(act, "detach") else act).reshape(-1, 14)
            executed = np.asarray(actions[step:step + PLAN], dtype=np.float32)
            n = min(len(executed), len(act))
            imgs = resp["images"]
            imgs = np.asarray(imgs.detach().cpu().float().numpy() if hasattr(imgs, "detach") else imgs)[0]  # C,T,H,W
            imagined.append(np.stack([to_u8(imgs[:, t]) for t in range(imgs.shape[1])][::12]))
            decisions.append({"decision": d, "step": step, "seed": seeds[d],
                              "action_max_abs_vs_executed": float(np.abs(act[:n] - executed[:n]).max()),
                              "action_mean_abs_vs_executed": float(np.abs(act[:n] - executed[:n]).mean()),
                              "value": float(np.asarray(resp["value"]).reshape(-1)[0]) if resp.get("value") is not None else None})
        env.take_action(np.asarray(actions[step], dtype=np.float32))
    ok = int(env.take_action_cnt) == a.stop_step and not env.eval_success
    env.close_env()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out.with_suffix(".real.npz"), frames=np.stack(real))
    np.savez_compressed(a.out.with_suffix(".imagined.npz"), keyframes=np.stack(imagined))
    a.out.with_suffix(".json").write_text(json.dumps({"rollout": str(a.rollout), "task": task, "env_seed": seed,
                                                      "stop_step": a.stop_step, "replay_ok": ok,
                                                      "decisions": decisions}, indent=1))
    print(json.dumps({"out": str(a.out), "decisions": len(decisions), "replay_ok": ok,
                      "max_action_diff": max(x["action_max_abs_vs_executed"] for x in decisions)}))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
