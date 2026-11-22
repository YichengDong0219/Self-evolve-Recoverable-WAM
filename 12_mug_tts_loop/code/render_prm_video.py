"""Check video: real frames of a round-0 TTS episode with the PRM's progress output (Claude, 2026-11-22). CPU only.
Frames: the executed trajectory every 12 steps = the current 3-view frame at each 48-step decision (online record)
plus the executed candidate's +12/+24/+36 keyframes (offline re-execution, bit-identical to the online run).
PRM values (Robo-Dopamine GRM-2.0-8B, forward mode, reference = the episode's first head frame, blank goal):
phi(cur_d) at every decision, and phi(+48 real) of the executed candidate; hop = incremental mode cur_{d-1} -> cur_d.
usage: python render_prm_video.py --round-dir <dir> --seeds 170002,170001 --out-dir <dir>
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PLAN = 48


def plot_img(steps_phi, phi, hop_steps, hop, t_now, total, w, h, success):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(steps_phi, phi, "o-", color="tab:blue", label="PRM progress phi (forward, vs start)")
    ax.bar(hop_steps, hop, width=20, color="tab:orange", alpha=0.6, label="PRM hop (incremental, prev->cur)")
    ax.axvline(t_now, color="k", lw=1)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlim(-10, max(total, 50) + 10); ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("step"); ax.set_ylabel("PRM output (fraction of 100%)")
    ax.set_title(f"episode outcome: {'SUCCESS' if success else 'FAILURE'} at step {total}", fontsize=9)
    ax.legend(loc="lower left", fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True); ap.add_argument("--seeds", required=True)
    ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--fps", type=int, default=4)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for seed in [int(s) for s in a.seeds.split(",")]:
        epf = a.round_dir / f"hanging_mug_seed{seed}.json"
        ep = json.loads(epf.read_text()); f = json.loads(epf.with_suffix(".prm.json").read_text())["features"]
        zn = np.load(epf.with_suffix(".tts.npz")); kz = np.load(epf.with_suffix(".exec.npz"))
        head, left, right, ks = zn["cur"], zn["cur_left"], zn["cur_right"], zn["k"]
        D, steps, succ = len(ks), ep["steps"], ep["success"]
        phi = [f["phi_cur"][str(d)] for d in range(D)]
        hop = [f["hop_prev"][str(d)] for d in range(1, D)]
        frames = []                                   # (step, [head, left, right], label)
        for d in range(D):
            frames.append((PLAN * d, [head[d], left[d], right[d]], f"decision {d}: PRM phi = {phi[d]:+.0%}"
                           + (f", hop = {hop[d - 1]:+.0%}" if d else "")))
            kk = f"d{d}_c{int(ks[d])}"
            if kk in kz.files:
                kf = kz[kk]
                n_kf = 4 if d == D - 1 else 3         # the +48 frame of decision d is decision d+1's frame
                for j in range(n_kf):
                    t = min(PLAN * d + 12 * (j + 1), steps)
                    lab = "between decisions (no PRM query)"
                    if j == 3:
                        lab = f"final, +48 real: PRM phi = {f['phi_re48'][f'{d}_{int(ks[d])}']:+.0%}"
                    frames.append((t, list(kf[j]), lab))
                    if t >= steps:
                        break
        H, W = head.shape[1:3]
        writer = None
        out = a.out_dir / f"prm_check_seed{seed}_{'success' if succ else 'failure'}.mp4"
        sp = [PLAN * d for d in range(D)]
        for t, (hd, lf, rt), lab in frames:
            wr = np.concatenate([cv2.resize(lf, (W // 2, H // 2)), cv2.resize(rt, (W // 2, H // 2))], 0)
            top = np.concatenate([hd, wr], 1)
            known = [i for i, s in enumerate(sp) if s <= t]
            pl = plot_img([sp[i] for i in known], [phi[i] for i in known], [sp[i] for i in known if i >= 1],
                          [hop[i - 1] for i in known if i >= 1], t, steps, top.shape[1], 300, succ)
            canvas = np.concatenate([top, pl], 0)
            bar = np.full((48, canvas.shape[1], 3), 255, np.uint8)
            cv2.putText(bar, f"seed {seed}  step {t}/{steps}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(bar, lab, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 0, 0), 1)
            canvas = np.concatenate([bar, canvas], 0)
            if writer is None:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (canvas.shape[1], canvas.shape[0]))
            reps = 3 if lab.startswith(("decision", "final")) else 1   # hold PRM-scored frames longer
            for _ in range(reps):
                writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        writer.release()
        print(out, len(frames), "frames")


if __name__ == "__main__":
    main()
