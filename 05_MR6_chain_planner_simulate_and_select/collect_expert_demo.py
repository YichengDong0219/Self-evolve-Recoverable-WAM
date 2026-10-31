"""Expert demonstrations on the scenes of a recovery set (Claude, 2026-10-28; task-generic 10-29). No model.
Used for turn_switch (E189) and hanging_mug (E arm of the mug comparison): same scene + instruction as the FACT rollout.

Control arm for the recovery distillation: for each env seed in MANIFEST_turn_switch_recovery_v1.jsonl, reset the same
scene with the same instruction text and run the stock RoboTwin expert (play_once) from the initial state. Frames are
recorded exactly like the expert segment of collect_recovery.py: three RGB cameras + 14-dim joint state through the
demonstration-collection hook (_take_picture every save_freq simulation steps), plus one frame before and after.
Outputs (stem --out): .json always; .npz (expert_rgb, expert_state) only on SUCCESS.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import g0_driver as G

CAMS = ("head_camera", "left_camera", "right_camera")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", type=Path, required=True, help="FACT rollout of this scene (seed, config, instruction)")
    ap.add_argument("--out", type=Path, required=True, help="output stem")
    a = ap.parse_args()
    if a.out.with_suffix(".json").exists():
        raise SystemExit("refusing to overwrite")
    src = json.loads(a.rollout.read_text())
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
    assert seed == src["env_seed"]
    G.setup_episode(env, args, seed, info, forced=src["instruction"])

    rgb, state = [], []

    def capture():  # same call sites and cadence as RoboTwin demonstration collection
        obs = env.get_obs()
        rgb.append(np.stack([np.asarray(obs["observation"][c]["rgb"][..., :3], dtype=np.uint8) for c in CAMS]))
        state.append(np.asarray(obs["joint_action"]["vector"], dtype=np.float32))

    env._take_picture = capture
    env.plan_success = True
    capture()
    t0, err = time.time(), None
    try:
        env.play_once()
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    capture()
    ok = bool(env.check_success())
    s = np.stack(state)
    disp = {"left": float(np.abs(s[:, :6] - s[0, :6]).max()), "right": float(np.abs(s[:, 7:13] - s[0, 7:13]).max())}
    out = {"format": "fact-expert-demo/1", "rollout": str(a.rollout), "task": task, "env_seed": seed,
           "instruction": src["instruction"], "status": "SUCCESS" if ok else "FAIL", "plan_success": bool(env.plan_success),
           "error": err, "expert_wall_s": round(time.time() - t0, 1), "expert_frames": len(rgb),
           "arm_disp": disp, "arm": max(disp, key=disp.get), "save_freq": env.save_freq, "camera_order": list(CAMS),
           "image_shape": list(rgb[0].shape[1:])}
    env.close_env()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if ok:
        np.savez_compressed(a.out.with_suffix(".npz"), expert_rgb=np.stack(rgb), expert_state=s)
    a.out.with_suffix(".json").write_text(json.dumps(out))
    print(json.dumps({x: out[x] for x in ("env_seed", "status", "error", "expert_frames", "arm")}))


if __name__ == "__main__":
    sys.exit(main())
