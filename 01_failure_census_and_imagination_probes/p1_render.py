"""Render a recorded episode to mp4 by bit-exact action replay (Claude, 2026-10-27). No model.

Frame = head | left wrist | right wrist cameras, with a text overlay (task, seed, step, outcome so far).
Rendering reads cameras only; the replayed outcome is re-checked against the source rollout.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import g0_driver as G

try:
    import cv2
except ImportError:  # overlay is optional
    cv2 = None


def frame(obs, text_lines):
    cams = [obs["observation"][k]["rgb"] for k in ("head_camera", "left_camera", "right_camera")]
    h = cams[0].shape[0]
    resized = []
    for c in cams:
        if c.shape[0] != h:
            w = int(round(c.shape[1] * h / c.shape[0]))
            c = cv2.resize(c, (w, h)) if cv2 is not None else c[:h]
        resized.append(c)
    img = np.ascontiguousarray(np.concatenate(resized, axis=1)[..., :3], dtype=np.uint8)
    if cv2 is not None:
        for i, t in enumerate(text_lines):
            cv2.putText(img, t, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, t, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
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
    G.setup_episode(env, args, seed, info, forced=src["instruction"],
                    override=src["instruction"] if src.get("instruction_source") == "override" else None)
    head = f"{task} seed {seed}  [{'SUCCESS' if src['success'] else 'FAIL'} @ {src['steps']}]"
    obs = env.get_obs()
    img = frame(obs, [head, "step 0"])
    tmp = a.out.with_suffix(".tmp.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{img.shape[1]}x{img.shape[0]}", "-r", str(a.fps), "-i", "-",
                           "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23", str(tmp)],
                          stdin=subprocess.PIPE)
    ff.stdin.write(img.tobytes())
    for i, act in enumerate(src["executed_actions"]):
        env.take_action(np.asarray(act, dtype=np.float32))
        obs = env.get_obs()
        g = env.robot
        line = f"step {env.take_action_cnt}  grip L {g.left_gripper_val:.2f} R {g.right_gripper_val:.2f}"
        ff.stdin.write(frame(obs, [head, line + ("  SUCCESS" if env.eval_success else "")]).tobytes())
        if env.eval_success:
            break
    ff.stdin.close()
    rc = ff.wait()
    ok = bool(env.eval_success) == src["success"] and int(env.take_action_cnt) == src["steps"]
    env.close_env()
    if rc != 0:
        raise SystemExit(f"ffmpeg rc={rc}")
    tmp.rename(a.out)
    print(json.dumps({"rollout": a.rollout.name, "video": str(a.out), "frames": int(env.take_action_cnt) + 1,
                      "replay_outcome_matches": ok}))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
