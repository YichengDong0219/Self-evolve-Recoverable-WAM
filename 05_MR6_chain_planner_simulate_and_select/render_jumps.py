"""Review clips of single-frame arm jumps in recovery segments (Claude, 2026-10-29). CPU only.

usage: python render_jumps.py <R_recovery.jsonl> <out_dir> [threshold=0.2]
For every recovery episode with a frame whose arm-joint step max|s[t+1]-s[t]| exceeds the threshold (expert data
max 0.13 rad), writes <out_dir>/<seed>.mp4: frames [t-25, t+25] around each jump at 5 fps (3x slower than real),
head camera 640x480 + both wrists. Banner per frame: frame index, that frame's arm step in rad; red on the jump
frame (t -> t+1) and the frame after it. Also writes ALL.mp4 (all clips in a row) and jumps.json.
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

ARM = [i for i in range(14) if i not in (6, 13)]


def tile(f3):
    return np.concatenate([cv2.resize(f3[0], (640, 480)), np.concatenate([f3[1], f3[2]], axis=0)], axis=1)


def banner(img, lines, color):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 24 * len(lines) + 8), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2 if i == 0 else 1, cv2.LINE_AA)
    return img


def write(frames, out, fps=5):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", out],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    p.wait()


def main():
    lst, od = sys.argv[1], sys.argv[2]
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    os.makedirs(od, exist_ok=False)
    allf, meta = [], []
    for line in open(lst):
        it = json.loads(line)
        z = np.load(it["npz"])
        st, rgb = z["expert_state"], z["expert_rgb"]
        d = np.abs(np.diff(st, axis=0))[:, ARM].max(axis=1)
        jumps = [int(t) for t in np.where(d > thr)[0]]
        if not jumps:
            continue
        m = it.get("action_masked_until")
        meta.append({"env_seed": it["env_seed"], "frames": len(st), "first_motion": m, "jumps": jumps,
                     "step_rad": [round(float(d[t]), 3) for t in jumps],
                     "joint_delta": [np.round(st[t + 1] - st[t], 3).tolist() for t in jumps]})
        clip = []
        for t in jumps:
            for k in range(max(0, t - 25), min(len(st), t + 26)):
                step = float(d[k]) if k < len(d) else 0.0
                hot = k in (t, t + 1)
                lines = [f"seed {it['env_seed']}  frame {k}/{len(st) - 1}  arm step {step:.3f} rad"
                         + ("   <<< JUMP" if k == t else ""),
                         f"jump at frame {t}->{t + 1}: {d[t]:.3f} rad (expert max 0.13); curobo first motion at frame {m}"]
                clip.append(banner(tile(rgb[k]), lines, (255, 60, 60) if hot else (255, 255, 255)))
            clip += [clip[-1]] * 5  # short hold between jumps
        write(clip, os.path.join(od, f"{it['env_seed']}.mp4"))
        allf += clip
        print(it["env_seed"], "jumps", jumps, [round(float(d[t]), 3) for t in jumps], flush=True)
    if allf:
        write(allf, os.path.join(od, "ALL.mp4"))
    json.dump(meta, open(os.path.join(od, "jumps.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
