"""Convert collected turn_switch trajectories (.npz) into a LeRobot v2.1 task directory for FACT SFT (Claude, 2026-10-28).

One episode per input file, built from the arrays <key>_rgb (T, 3, H, W, 3; head, left wrist, right wrist) and
<key>_state (T, 14). Convention verified on FACT's own executions (action_t == state_{t+1} exactly):
  observation.state[t] = state[t],  action[t] = state[t+1]  (last frame: action = state, i.e. hold)
Used for both SFT arms so they differ only in the episodes:
  E200 (expert demos, key "expert")  and  R202 (expert segments of the recovery set, key "expert").
Layout written under --out (one RoboTwin task dir; FACT discovers it through meta/info.json):
  meta/{info.json, episodes.jsonl, tasks.jsonl, sources.jsonl}
  data/chunk-000/episode_XXXXXX.parquet
  videos/chunk-000/observation.images.cam_{high,left_wrist,right_wrist}/episode_XXXXXX.mp4   (H.264, crf 10)
T5 embeddings and VAE latents are added afterwards by FACT's own scripts.
Optional per-item fields (10-29, hanging_mug recovery arm): "end" keeps frames [0, end] only (failure prefix truncated
where the falling mug comes to rest); "failure_active_from" marks the episode as a FACT failure episode and writes
meta/failure_rollouts.jsonl {episode_index, failure_episode: true, failure_active_from_frame} (action loss masked for
the whole episode, value penalty from that frame on). "action_masked_until" m (recovery segments, 10-29) writes a row
{failure_episode: false, action_masked_until_frame: m}: no action loss on frames < m (curobo still waiting for the
mug to settle), handled by action_mask_patch.py; plain FACT ignores it.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 15
VIEWS = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
FF = os.environ.get("FFMPEG") or __import__("shutil").which("ffmpeg") or "/workspace_cluster/miniconda3/bin/ffmpeg"


def write_video(frames, path):
    t, h, w, _ = frames.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p",
                          str(path)], stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(frames, dtype=np.uint8).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"ffmpeg failed for {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, required=True, help="jsonl with fields npz, instruction, env_seed")
    ap.add_argument("--key", default="expert", help="array prefix inside the npz")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if (a.out / "meta" / "info.json").exists():
        raise SystemExit("refusing to overwrite")
    items = [json.loads(line) for line in a.list.read_text().splitlines() if line.strip()]
    tasks, episodes, sources, failures, index = {}, [], [], [], 0
    shape = None
    for ep, it in enumerate(items):
        z = np.load(it["npz"])
        rgb, state = z[f"{a.key}_rgb"], z[f"{a.key}_state"].astype(np.float32)
        if it.get("end") is not None:
            rgb, state = rgb[: int(it["end"]) + 1], state[: int(it["end"]) + 1]
        T = len(state)
        assert rgb.shape[0] == T and rgb.shape[1] == 3, (it["npz"], rgb.shape)
        shape = list(rgb.shape[2:])
        action = np.concatenate([state[1:], state[-1:]], axis=0)
        ti = tasks.setdefault(it["instruction"], len(tasks))
        df = pd.DataFrame({
            "observation.state": list(state), "action": list(action),
            "timestamp": np.arange(T, dtype=np.float32) / FPS, "frame_index": np.arange(T, dtype=np.int64),
            "episode_index": np.full(T, ep, dtype=np.int64), "index": np.arange(index, index + T, dtype=np.int64),
            "task_index": np.full(T, ti, dtype=np.int64)})
        p = a.out / f"data/chunk-000/episode_{ep:06d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)
        for v, key in enumerate(VIEWS):
            write_video(rgb[:, v], a.out / f"videos/chunk-000/{key}/episode_{ep:06d}.mp4")
        episodes.append({"episode_index": ep, "tasks": [it["instruction"]], "length": T})
        sources.append({"episode_index": ep, "env_seed": it["env_seed"], "npz": it["npz"], "length": T})
        if it.get("failure_active_from") is not None:
            failures.append({"episode_index": ep, "failure_episode": True,
                             "failure_active_from_frame": int(it["failure_active_from"])})
        elif it.get("action_masked_until") is not None:  # read by action_mask_patch.py (per-frame action mask)
            failures.append({"episode_index": ep, "failure_episode": False,
                             "action_masked_until_frame": int(it["action_masked_until"])})
        index += T
        print(ep, it["env_seed"], T, flush=True)
    vid = {"dtype": "video", "shape": shape, "names": ["height", "width", "channel"],
           "info": {"video.fps": FPS, "video.codec": "h264", "video.pix_fmt": "yuv420p"}}
    info = {"codebase_version": "v2.1", "robot_type": "aloha", "fps": FPS, "total_episodes": len(episodes),
            "total_frames": index, "total_tasks": len(tasks), "total_videos": 3 * len(episodes), "total_chunks": 1,
            "chunks_size": 1000, "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {**{k: vid for k in VIEWS},
                         "observation.state": {"dtype": "float32", "shape": [14], "names": None},
                         "action": {"dtype": "float32", "shape": [14], "names": None},
                         "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                         "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                         "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                         "index": {"dtype": "int64", "shape": [1], "names": None},
                         "task_index": {"dtype": "int64", "shape": [1], "names": None}}}
    meta = a.out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}) + "\n" for t, i in tasks.items()))
    (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (meta / "sources.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sources))
    (meta / "info.json").write_text(json.dumps(info, indent=1))
    if failures:
        (meta / "failure_rollouts.jsonl").write_text("".join(json.dumps(f) + "\n" for f in failures))
    print(json.dumps({"episodes": len(episodes), "frames": index, "tasks": len(tasks)}))


if __name__ == "__main__":
    main()
