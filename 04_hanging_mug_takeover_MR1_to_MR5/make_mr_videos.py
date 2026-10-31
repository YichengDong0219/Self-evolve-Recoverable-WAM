"""Failure + recovery videos for the hanging_mug task-specific recovery benchmark (Claude, 2026-10-28). CPU only.

usage: python make_mr_videos.py <bench variant dir, e.g. bench/MRv4>
For each attempt: FACT policy part (original rollout video, steps 0..k, one frame per step; IMG videos are cropped to
the real cameras) followed by the recovery segment recorded by takeover_bench.py --video (demonstration cadence),
both at 25 fps, 960x240 (head | left wrist | right wrist), with a label bar per phase.
Output: videos/<variant>/<success|fail>/<group>_seed<seed>_<class>_<RESULT>.failure-plus-recovery.mp4
"""
import glob
import json
import os
import subprocess
import sys

L = "/workspace/fact_local"


def main():
    vdir = sys.argv[1].rstrip("/")
    variant = os.path.basename(vdir)
    n = 0
    for j in sorted(glob.glob(f"{vdir}/*.json")):
        d = json.load(open(j))
        rec = j[:-5] + ".mp4"
        if not os.path.exists(rec):
            continue
        roll = os.path.basename(d["rollout"])[:-5]
        group = "P1" if roll.startswith("P1_") else "IMG"
        if group == "P1":
            src, crop = f"{L}/orig_videos/{roll}.mp4", ""
        else:
            src = glob.glob(f"{L}/orig_videos/IMG_hanging_mug_seed{d['env_seed']}_FAIL_*.mp4")[0]
            crop = "crop=960:240:0:0,"
        st = {e["stage"]: e for e in (d.get("mr_log") or [])}
        c = (st.get("parked") or st.get("classify") or {})
        cls = f'{c.get("class", "?")}{"-near-rack" if c.get("near_rack") else ""}'
        res = d["status"]
        k = d["step"]
        dst = f"{L}/videos/{variant}/{'success' if res == 'SUCCESS' else 'fail'}"
        os.makedirs(dst, exist_ok=True)
        out = f"{dst}/{group}_seed{d['env_seed']}_{cls}_{res}.failure-plus-recovery.mp4"
        if os.path.exists(out):
            continue
        rs = st.get("restage", {})
        lab2 = (f"EXPERT RECOVERY {variant} - settle, park, restage ({rs.get('outcome', '?')}"
                f"{', middle_pos' if rs.get('spot') == [0.0, -0.15] else ''}) then stock hang -> {res}")
        paths = [e for e in (d.get("mr_log") or []) if e["stage"] == "path"]
        if paths:  # MR5: direct pick-and-hang from the fallen pose, no homing
            lab2 = f"EXPERT RECOVERY {variant} - settle, pick where it fell, hang directly (no homing, {paths[-1]['path']}) -> {res}"
        lab2 = lab2.replace(",", "\\,")  # drawtext: ',' and ':' are filter syntax
        flt = (f"[0:v]{crop}select='lte(n\\,{k})',setpts=N/25/TB,"
               f"drawtext=text='FACT POLICY steps 0-{k} (failure state - {cls})':x=6:y=222:fontsize=14:fontcolor=yellow:box=1:boxcolor=black@0.6[a];"
               f"[1:v]fps=25,setpts=N/25/TB,"
               f"drawtext=text='{lab2}':x=6:y=222:fontsize=13:fontcolor=cyan:box=1:boxcolor=black@0.6[b];"
               f"[a][b]concat=n=2:v=1:a=0[v]")
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", src, "-i", rec, "-filter_complex", flt, "-map", "[v]",
                            "-r", "25", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "23", out],
                           capture_output=True, text=True)
        if r.returncode:
            print("FFMPEG_FAIL", j, r.stderr[-300:])
            continue
        link = out.replace(".failure-plus-recovery.mp4", ".recovery-segment-only.mp4")
        if not os.path.exists(link):
            os.link(rec, link)
        n += 1
        print(res, os.path.basename(out))
    print("videos:", n)


if __name__ == "__main__":
    main()
