"""Failure + recovery videos for stage-aware dispatch trials (Claude, 2026-10-27). CPU only (ffmpeg).

Same layout as make_takeover_videos.py, for ~/fact_takeover/dispatch (hanging_mug L1/R1/R2) and
~/fact_takeover/dispatch_multi (turn_switch W1/W2, move_stapler_pad S1/S2). Attempts whose plan was not applicable
executed no motion and are skipped. Output: fact_results/08_DSP_stage-aware-dispatch-smoke/<task>/<success|fail>/.
"""
import glob
import json
import os
import subprocess

FF = "/workspace_cluster/miniconda3/bin/ffmpeg"
P1V = "/workspace_cluster/fact_g0_claude/runs/videos"
OUT = "/workspace_cluster/fact_results/08_DSP_stage-aware-dispatch-smoke"
NAMES = {"L1": "left-restart", "R1": "right-to-middle-then-handover", "R2": "right-to-start-then-play_once",
         "W1": "press-preferred-arm", "W2": "press-other-arm", "S1": "regrasp-place-preferred-arm",
         "S2": "regrasp-place-other-arm"}

rows = []
for d_dir in ("/workspace_cluster/fact_takeover/dispatch", "/workspace_cluster/fact_takeover/dispatch_multi"):
    for j in sorted(glob.glob(f"{d_dir}/*_dispatch*.json")):
        d = json.load(open(j))
        if d["status"] == "not_applicable":
            continue
        stem = os.path.basename(j)[:-5]
        src = stem.split("_step")[0]
        k, plan = d["branch_step"], d["plan"]
        task = d.get("task", "hanging_mug")
        res = d["status"]
        base = f"DSP_{task}_seed{d['env_seed']}_P1-fail_dispatch-from-step{k:03d}_{plan}-{NAMES[plan]}_{res}"
        dst = f"{OUT}/{task}/{res.lower()}"
        os.makedirs(dst, exist_ok=True)
        combined = f"{dst}/{base}.failure-plus-recovery.mp4"
        if not os.path.exists(combined):
            flt = (f"[0:v]select='lte(n\\,{k})',setpts=N/25/TB,"
                   f"drawtext=text='POLICY (FACT) steps 0-{k}':x=6:y=220:fontsize=14:fontcolor=yellow:box=1:boxcolor=black@0.6[a];"
                   f"[1:v]fps=25,setpts=N/25/TB,"
                   f"drawtext=text='EXPERT DISPATCH {plan} ({NAMES[plan]}) -> {res}':x=6:y=220:fontsize=14:fontcolor=cyan:box=1:boxcolor=black@0.6[b];"
                   f"[a][b]concat=n=2:v=1:a=0[v]")
            r = subprocess.run([FF, "-loglevel", "error", "-i", f"{P1V}/{src}.mp4", "-i", f"{d_dir}/{stem}.mp4",
                                "-filter_complex", flt, "-map", "[v]", "-r", "25", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                                "-crf", "23", combined], capture_output=True, text=True)
            if r.returncode:
                print("FFMPEG_FAIL", stem, r.stderr[-400:], flush=True)
                continue
        for s, suf in ((f"{d_dir}/{stem}.mp4", ".expert-segment-only.mp4"), (j, ".dispatch.json"),
                       (f"{P1V}/{src}.mp4", ".original-P1-failure.mp4")):
            t = f"{dst}/{base}{suf}"
            if not os.path.exists(t):
                os.link(s, t)
        rows.append((task, d["env_seed"], plan, res))
        print(task, d["env_seed"], plan, res, flush=True)
print("videos:", len(rows))
