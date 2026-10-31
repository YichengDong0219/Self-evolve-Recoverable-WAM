"""Review videos of the SFT-ready hanging_mug failure-recovery pairs (Claude, 2026-10-29). CPU only.

usage: python render_sft_ready.py <list_dir from mug_sft_lists.py> <out_dir> [workers=16]
One mp4 per pair, rendered from exactly the frames that go into SFT (capture npz), 15 fps (policy step rate):
  part 1  FACT failure prefix, frames 0..end (end = mug at rest). Yellow banner = before the fall onset;
          red banner = failure-active frames (value penalty). Action loss is off for the whole prefix.
  part 2  curobo recovery from the takeover state (settle frames at the start included), cyan banner
          (action-supervised in SFT).
  part 3  the expert demo of the same scene (green banner, E arm) for comparison.
Layout per frame: head camera 640x480 on the left, both wrist cameras stacked at 320x240 on the right.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

FPS = 15


def tile(f3):
    head = cv2.resize(f3[0], (640, 480), interpolation=cv2.INTER_LINEAR)
    right = np.concatenate([f3[1], f3[2]], axis=0)
    return np.concatenate([head, right], axis=1)


def banner(img, text, color, sub=""):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 30 if not sub else 52), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def render(args):
    rec, pre, exp, out = args
    if os.path.exists(out):
        return out, "exists"
    z = np.load(rec["npz"])
    e = np.load(exp["npz"])
    end, on = pre["end"], pre["failure_active_from"]
    YEL, RED, CYA, GRN = (255, 220, 0), (255, 60, 60), (0, 220, 255), (80, 255, 80)  # RGB
    sub = f"seed {rec['env_seed']}: {rec['instruction'][:90]}"
    frames = []
    P = z["prefix_rgb"]
    for t in range(end + 1):
        if t < on:
            frames.append(banner(tile(P[t]), f"1/3 FACT policy  t={t}/{end}  (failure prefix, no action loss)", YEL, sub))
        else:
            frames.append(banner(tile(P[t]), f"1/3 FACT policy  t={t}/{end}  FALL: value penalty from t={on}", RED, sub))
    X = z["expert_rgb"]
    for t in range(len(X)):
        frames.append(banner(tile(X[t]), f"2/3 curobo RECOVERY  {t}/{len(X) - 1}  (action-supervised; starts with settle)",
                             CYA, f"takeover at FACT step {len(P)}; prefix shown up to rest t={end}"))
    E = e["expert_rgb"]
    for t in range(len(E)):
        frames.append(banner(tile(E[t]), f"3/3 EXPERT demo, same scene  {t}/{len(E) - 1}  (E arm)", GRN, sub))
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", out + ".tmp.mp4"],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    if p.wait():
        return out, "ffmpeg failed"
    os.rename(out + ".tmp.mp4", out)
    return out, f"{end + 1}+{len(X)}+{len(E)} frames"


def main():
    ld, od = sys.argv[1], sys.argv[2]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    os.makedirs(od, exist_ok=True)
    rd = lambda n: [json.loads(x) for x in open(os.path.join(ld, n))]
    R, Pf, E = rd("R_recovery.jsonl"), rd("R_prefix.jsonl"), rd("E_expert.jsonl")
    names = json.load(open(os.path.join(ld, "paired.json")))["names"]
    jobs = [(r, p, e, os.path.join(od, f"{i:03d}_{n}_seed{r['env_seed']}.mp4")) for i, (n, r, p, e) in enumerate(zip(names, R, Pf, E))]
    with ProcessPoolExecutor(nw) as ex:
        for out, msg in ex.map(render, jobs):
            print(os.path.basename(out), msg, flush=True)


if __name__ == "__main__":
    main()
