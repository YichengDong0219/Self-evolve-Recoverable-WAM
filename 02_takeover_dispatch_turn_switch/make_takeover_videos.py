"""Failure + recovery videos for takeover smoke trials (Claude, 2026-10-27). CPU only (ffmpeg).

For each takeover trial: cut the P1 failure video (one frame per policy step; frame k = state after step k) at the
takeover step, then append the expert takeover segment (one frame per 15 simulation steps, re-timed to 25 fps), with
a phase label burned in. Output: fact_results/07_TKO_expert-takeover-smoke/<task>/<success|fail>/ (never overwrites).
"""
import glob
import json
import os
import subprocess

FF = "/workspace_cluster/miniconda3/bin/ffmpeg"
SMOKE = "/workspace_cluster/fact_takeover/smoke"
P1V = "/workspace_cluster/fact_g0_claude/runs/videos"
OUT = "/workspace_cluster/fact_results/07_TKO_expert-takeover-smoke"
NAMES = {"A": "stock-play_once", "B": "right-arm-direct-hang", "C": "restage-then-play_once"}

rows = []
for j in sorted(glob.glob(f"{SMOKE}/*_takeover?.json")):
    d = json.load(open(j))
    stem = os.path.basename(j)[:-5]                      # P1_<task>_demo_clean_NN_step<k>_takeover<S>
    src = stem.split("_step")[0]
    k = d["branch_step"]
    task, seed, st = d.get("task", "hanging_mug"), d["env_seed"], d["strategy"]
    res = "SUCCESS" if d["success"] else "FAIL"
    base = f"TKO_{task}_seed{seed}_P1-fail_expert-from-step{k:03d}_{NAMES[st]}_{res}"
    dst = f"{OUT}/{task}/{res.lower()}"
    os.makedirs(dst, exist_ok=True)
    combined = f"{dst}/{base}.failure-plus-recovery.mp4"
    if not os.path.exists(combined):
        flt = (f"[0:v]select='lte(n\\,{k})',setpts=N/25/TB,"
               f"drawtext=text='POLICY (FACT) steps 0-{k}':x=6:y=220:fontsize=14:fontcolor=yellow:box=1:boxcolor=black@0.6[a];"
               f"[1:v]fps=25,setpts=N/25/TB,"
               f"drawtext=text='EXPERT TAKEOVER ({NAMES[st]}) -> {res}':x=6:y=220:fontsize=14:fontcolor=cyan:box=1:boxcolor=black@0.6[b];"
               f"[a][b]concat=n=2:v=1:a=0[v]")
        r = subprocess.run([FF, "-loglevel", "error", "-i", f"{P1V}/{src}.mp4", "-i", f"{SMOKE}/{stem}.mp4",
                            "-filter_complex", flt, "-map", "[v]", "-r", "25", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                            "-crf", "23", combined], capture_output=True, text=True)
        if r.returncode:
            print("FFMPEG_FAIL", stem, r.stderr[-400:], flush=True)
            continue
    for s, suf in ((f"{SMOKE}/{stem}.mp4", ".expert-segment-only.mp4"), (f"{SMOKE}/{stem}.json", ".takeover.json"),
                   (f"{P1V}/{src}.mp4", ".original-P1-failure.mp4")):
        t = f"{dst}/{base}{suf}"
        if not os.path.exists(t):
            os.link(s, t)
    rows.append((task, seed, k, st, res))
    print(task, seed, k, st, res, flush=True)
print("videos:", len(rows))
