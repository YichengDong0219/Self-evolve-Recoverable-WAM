"""Reconstruct FACT's imagined futures along a recorded episode and render real-vs-imagined video (Claude, 2026-10-27).

The recorded rollout ran the deployment path (action-only, prefix cache on), so no imagination was stored.
Here the episode is replayed bit-exactly; at every decision boundary (step % 48 == 0) the identical request
(same three camera images, joint state, instruction, and the ORIGINAL explicit seed of that decision) is sent to a
probe server started with --return_images. That path cannot use the prefix cache, so its action may differ
slightly from the executed one; the per-decision difference is reported, so the imagination's fidelity to what the
model "had in mind" can be judged. Video: top = real head | left | right cameras; bottom = imagined canvas
(4 predicted keyframes per decision, each held for 12 steps, exactly as the server expands them).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import g0_driver as G

PLAN = 48


def to_u8(img_chw_m11):
    x = np.asarray(img_chw_m11, dtype=np.float32)
    x = np.transpose(x, (1, 2, 0))
    return np.clip((x + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--out", type=Path, required=True, help="output .mp4; a .json/.npz with the same stem is written too")
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    import cv2
    from evaluation.robotwin.model2robotwin_interface import _to_chw01
    from world_action_model.sockets import RobotInferenceClient

    src = json.loads(a.rollout.read_text())
    seeds = [r["probe_seeds"] for r in src["server_io"] if r["kind"] == "response"]
    task = src["task"]
    args = G.build_args(task, src["config"])
    env = G.task_env(task)
    for attempt in range(1, 4):
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

    W, H = 960, 240
    tmp = a.out.with_suffix(".tmp.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H * 2}", "-r", str(a.fps), "-i", "-", "-pix_fmt", "yuv420p",
                           "-vcodec", "libx264", "-crf", "23", str(tmp)], stdin=subprocess.PIPE)
    decisions, imagined_store = [], []
    imagined = None
    actions = src["executed_actions"]

    def label(img, lines, color=(255, 255, 255)):
        for i, t in enumerate(lines):
            cv2.putText(img, t, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, t, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return img

    head = f"{task} seed {seed}  [{'SUCCESS' if src['success'] else 'FAIL'} @ {src['steps']}]"
    step = 0
    while True:
        obs = env.get_obs()
        if step % PLAN == 0 and step < len(actions):
            d = step // PLAN
            cams = [obs["observation"][k]["rgb"] for k in ("head_camera", "left_camera", "right_camera")]
            req = {"observation.state": np.asarray(obs["joint_action"]["vector"], dtype=np.float32).reshape(-1),
                   "observation.images.cam_high": _to_chw01(cams[0]),
                   "observation.images.cam_left_wrist": _to_chw01(cams[1]),
                   "observation.images.cam_right_wrist": _to_chw01(cams[2]),
                   "instruction": src["instruction"], "best_of_N": 1, "probe_seeds": list(seeds[d])}
            resp = client.inference(req)
            act = resp["action"]
            act = np.asarray(act.detach().cpu().float().numpy() if hasattr(act, "detach") else act).reshape(-1, 14)
            executed = np.asarray(actions[step:step + PLAN], dtype=np.float32)
            n = min(len(executed), len(act))
            imgs = resp["images"]
            imgs = np.asarray(imgs.detach().cpu().float().numpy() if hasattr(imgs, "detach") else imgs)[0]  # C,T,H,W
            imagined = [to_u8(imgs[:, t]) for t in range(imgs.shape[1])]
            imagined_store.append(np.stack(imagined[::12]))  # the distinct keyframes
            decisions.append({"decision": d, "step": step, "seed": seeds[d],
                              "action_max_abs_vs_executed": float(np.abs(act[:n] - executed[:n]).max()),
                              "action_mean_abs_vs_executed": float(np.abs(act[:n] - executed[:n]).mean()),
                              "value": float(np.asarray(resp["value"]).reshape(-1)[0]) if resp.get("value") is not None else None})
        real = np.concatenate([cv2.resize(c, (320, 240)) if c.shape[:2] != (240, 320) else c
                               for c in (obs["observation"][k]["rgb"][..., :3]
                                         for k in ("head_camera", "left_camera", "right_camera"))], axis=1)
        real = label(np.ascontiguousarray(real, dtype=np.uint8),
                     [head, f"REAL  step {step}" + ("  SUCCESS" if env.eval_success else "")])
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        if imagined is not None:
            k = min(step % PLAN, len(imagined) - 1) if step < len(actions) else len(imagined) - 1
            im = imagined[k]
            s = min(W / im.shape[1], H / im.shape[0])
            im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
            x0, y0 = (W - im.shape[1]) // 2, (H - im.shape[0]) // 2
            canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            dd = decisions[-1]
            label(canvas, [f"IMAGINED (decision {dd['decision']} @ step {dd['step']}, keyframe {k // 12 + 1}/4)",
                           f"regenerated action vs executed: max {dd['action_max_abs_vs_executed']:.3f}"],
                  color=(255, 220, 120))
        ff.stdin.write(np.concatenate([real, canvas], axis=0).tobytes())
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
    np.savez_compressed(a.out.with_suffix(".imagined.npz"), keyframes=np.stack(imagined_store))
    a.out.with_suffix(".json").write_text(json.dumps({"rollout": str(a.rollout), "replay_outcome_matches": ok,
                                                      "decisions": decisions}, indent=1))
    print(json.dumps({"video": str(a.out), "decisions": len(decisions), "replay_outcome_matches": ok,
                      "max_action_diff": max(x["action_max_abs_vs_executed"] for x in decisions)}))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
