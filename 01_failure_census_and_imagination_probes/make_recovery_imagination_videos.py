"""Failure + recovery videos with FACT's imagination, in a larger viewing layout (Claude, 2026-10-27). CPU only.

For every SUCCESS (or, with --status fail / all, every executed FAIL) in 07_TKO / 08_DSP: left column = real cameras, right column = what FACT imagined at that moment
(reconstructed by imagine_prefix.py). Each column: head camera 640x480 on top, left | right wrist 320x240 below.
During the expert segment FACT is not queried; the right column shows the last imagination dimmed.
Output: fact_results/09_IMR_recovery-with-imagination/<07_TKO|08_DSP>/<task>/<stem>.real-vs-imagined.mp4
"""
import argparse
import glob
import json
import os
import subprocess

import cv2
import numpy as np

FF = "/workspace_cluster/miniconda3/bin/ffmpeg"
RES = "/workspace_cluster/fact_results"
PRE = "/workspace_cluster/fact_takeover/imagine_prefix"
OUT = f"{RES}/09_IMR_recovery-with-imagination"
W, BAR, PH = 1280, 40, 720
YEL, CYA, WHT = (255, 210, 60), (80, 220, 255), (255, 255, 255)


def text(img, s, xy, color=WHT, scale=0.6):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def column(head, left, right, title, color):
    c = np.zeros((PH, 640, 3), np.uint8)
    c[:480] = cv2.resize(head, (640, 480), interpolation=cv2.INTER_CUBIC)
    c[480:, :320] = cv2.resize(left, (320, 240), interpolation=cv2.INTER_CUBIC)
    c[480:, 320:] = cv2.resize(right, (320, 240), interpolation=cv2.INTER_CUBIC)
    cv2.line(c, (0, 480), (640, 480), (0, 0, 0), 2)
    cv2.line(c, (320, 480), (320, 720), (0, 0, 0), 2)
    text(c, title, (8, 24), color)
    text(c, "head", (8, 470), WHT, 0.5)
    text(c, "left wrist", (8, 710), WHT, 0.5)
    text(c, "right wrist", (328, 710), WHT, 0.5)
    return c


def split_imagined(kf):  # FACT canvas: head 256x192 left, left wrist top-right, right wrist bottom-right
    return kf[:, :256], kf[:96, 256:], kf[96:, 256:]


def read_video(path):
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, fr = cap.read()
        if not ok:
            return out
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))


def published_base(src_json):  # 07/08 trees hold hard links of the source json; match by inode
    ino = os.stat(src_json).st_ino
    for p in glob.glob(f"{RES}/0[78]_*/*/*/*.json"):
        if os.stat(p).st_ino == ino:
            return p, p.rsplit(".", 2)[0]
    raise SystemExit(f"no published video for {src_json}")


def render(stem, d, src_dir, batch):
    name = os.path.basename(d["rollout"])[:-5]
    task, seed, k = name[3:name.index("_demo_")], d["env_seed"], d["branch_step"]
    meta = json.load(open(f"{PRE}/{name}_k{k}.json"))
    real = np.load(f"{PRE}/{name}_k{k}.real.npz")["frames"]
    imag = np.load(f"{PRE}/{name}_k{k}.imagined.npz")["keyframes"]
    expert = read_video(f"{src_dir}/{os.path.basename(stem)}.mp4")
    who = d.get("plan") or d.get("strategy") or "expert"
    pub, base = published_base(f"{stem}.json")
    result = os.path.basename(os.path.dirname(pub)).upper()
    dst = f"{OUT}/{batch}/{task}" + ("" if result == "SUCCESS" else "/fail")
    os.makedirs(dst, exist_ok=True)
    out = f"{dst}/{os.path.basename(base)}.real-vs-imagined.mp4"
    if os.path.exists(out):
        return out
    ff = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{BAR + PH}",
                           "-r", "25", "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "20", out],
                          stdin=subprocess.PIPE)
    last = None
    for step in range(k + 1):
        di = min(step // 48, len(imag) - 1)
        j = min((step % 48) // 12, 3) if step < k else 3
        last = imag[di][j]
        dec = meta["decisions"][di]
        fr = np.zeros((BAR + PH, W, 3), np.uint8)
        fr[:BAR] = (60, 50, 0)
        text(fr, f"{task}  seed {seed}   POLICY (FACT)  step {step}/{k}   -> expert takeover at step {k}", (10, 27), YEL)
        fr[BAR:, :640] = column(*real[step], "REAL", YEL)
        fr[BAR:, 640:] = column(*split_imagined(last),
                                f"FACT IMAGINED  decision {dec['decision']} @ step {dec['step']}  keyframe {j + 1}/4", YEL)
        cv2.rectangle(fr, (0, 0), (W - 1, BAR + PH - 1), YEL, 3)
        ff.stdin.write(fr.tobytes())
    dim = (np.asarray(last, np.float32) * 0.35).astype(np.uint8)
    for i, e in enumerate(expert):  # expert video: head | left | right strip, 320x240 each, small label at top-left
        cams = [e[:, 320 * c:320 * (c + 1)] for c in range(3)]
        fr = np.zeros((BAR + PH, W, 3), np.uint8)
        fr[:BAR] = (0, 50, 70)
        text(fr, f"{task}  seed {seed}   EXPERT TAKEOVER ({who})  frame {i + 1}/{len(expert)}   result: {result}", (10, 27), CYA)
        fr[BAR:, :640] = column(*cams, "REAL", CYA)
        col = column(*split_imagined(dim), "FACT not queried during expert takeover", CYA)
        text(col, "(last imagination, dimmed)", (8, 50), CYA, 0.5)
        fr[BAR:, 640:] = col
        cv2.rectangle(fr, (0, 0), (W - 1, BAR + PH - 1), CYA, 3)
        ff.stdin.write(fr.tobytes())
    ff.stdin.close()
    if ff.wait():
        raise SystemExit(f"ffmpeg failed for {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", choices=["success", "fail", "all"], default="success")
    a = ap.parse_args()
    jobs = []
    for batch, src_dir in (("07_TKO", "/workspace_cluster/fact_takeover/smoke"),
                           ("08_DSP", "/workspace_cluster/fact_takeover/dispatch"),
                           ("08_DSP", "/workspace_cluster/fact_takeover/dispatch_multi")):
        for j in sorted(glob.glob(f"{src_dir}/*.json")):
            d = json.load(open(j))
            if "takeover" not in j and "dispatch" not in j:
                continue
            ok = d.get("status") == "SUCCESS" or d.get("success") is True
            ran = d.get("status") != "not_applicable" and os.path.exists(j[:-5] + ".mp4")
            if ran and (a.status == "all" or (a.status == "success") == ok):
                jobs.append((j[:-5], d, src_dir, batch))
    for stem, d, src_dir, batch in jobs:
        print(render(stem, d, src_dir, batch), flush=True)
    print("videos:", len(jobs))


if __name__ == "__main__":
    main()
