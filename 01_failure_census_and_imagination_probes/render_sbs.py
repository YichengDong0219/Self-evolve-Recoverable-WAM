"""Side-by-side real vs imagined video for a rollout recorded with a --return_images server (Claude, 2026-10-27).

Left: real head | left wrist | right wrist cameras (bit-exact action replay, as validated in G0).
Right: the imagination that was generated in the SAME forward pass as the executed chunk (stored keyframes,
4 per decision, keyframe i shown on steps 12i..12i+11 of that chunk, exactly as the FACT server expands them).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

import g0_driver as G

PLAN = 48


def label(img, lines, color=(255, 255, 255)):
    for i, t in enumerate(lines):
        cv2.putText(img, t, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
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
    keys = np.load(a.rollout.with_suffix(".imagined.npz"))["keyframes"]  # D, 4, H, W, 3
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
    H, RW, IW = 240, 960, 480
    head = f"{task} seed {seed}  [{'SUCCESS' if src['success'] else 'FAIL'} @ {src['steps']}]"
    tmp = a.out.with_suffix(".tmp.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{RW + IW}x{H}", "-r", str(a.fps), "-i", "-", "-pix_fmt", "yuv420p",
                           "-vcodec", "libx264", "-crf", "23", str(tmp)], stdin=subprocess.PIPE)
    actions = src["executed_actions"]
    step = 0
    while True:
        obs = env.get_obs()
        cams = [obs["observation"][k]["rgb"][..., :3] for k in ("head_camera", "left_camera", "right_camera")]
        real = np.ascontiguousarray(np.concatenate([cv2.resize(c, (320, 240)) for c in cams], axis=1), dtype=np.uint8)
        label(real, [head, f"REAL  step {step}" + ("  SUCCESS" if env.eval_success else "")])
        d = min(step // PLAN, len(keys) - 1)
        k = min((step % PLAN) // 12, keys.shape[1] - 1) if step < len(actions) else keys.shape[1] - 1
        im = np.ascontiguousarray(cv2.resize(keys[d, k], (IW, int(round(keys.shape[2] * IW / keys.shape[3])))))
        pane = np.zeros((H, IW, 3), dtype=np.uint8)
        y0 = (H - im.shape[0]) // 2
        pane[y0:y0 + im.shape[0]] = im[:H]
        label(pane, [f"IMAGINED  decision {d} (step {d * PLAN})", f"keyframe {k + 1}/4"], color=(255, 220, 120))
        ff.stdin.write(np.concatenate([real, pane], axis=1).tobytes())
        if step >= len(actions) or env.eval_success:
            break
        env.take_action(np.asarray(actions[step], dtype=np.float32))
        step += 1
    ff.stdin.close()
    rc = ff.wait()
    ok = bool(env.eval_success) == src["success"] and int(env.take_action_cnt) == src["steps"]
    env.close_env()
    if rc != 0:
        raise SystemExit(f"ffmpeg rc={rc}")
    tmp.rename(a.out)
    print(json.dumps({"rollout": a.rollout.name, "video": str(a.out), "decisions": int(len(keys)),
                      "replay_outcome_matches": ok}))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
