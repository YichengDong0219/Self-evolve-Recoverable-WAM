"""Index of rendered hanging_mug eval videos (Claude, 2026-10-31). usage: python eval_video_index.py <video dir>

Writes INDEX.tsv (one row per seed: outcome + video per checkpoint), M3000_vs_E3000.tsv (seeds where the two disagree)
and README.txt. Outcomes come from the video names; replay checks from the render logs."""
import glob
import json
import os
import re
import sys

TAGS = ["mugM_nv_step1000", "mugM_nv_step2000", "mugM_nv_step3000", "mugE_nv_step3000"]


def main(d):
    res, bad = {}, []
    for t in TAGS:
        for f in glob.glob(f"{d}/{t}/*.mp4"):
            mt = re.fullmatch(r"hanging_mug_seed(\d+)_(SUCCESS|FAIL)\.mp4", os.path.basename(f))
            if not mt:  # .tmp.mp4 of a render still in progress
                continue
            s, o = mt.groups()
            res.setdefault(int(s), {})[t] = (o, os.path.relpath(f, d))
        for lg in glob.glob(f"{d}/{t}/*.log"):
            last = open(lg).read().strip().splitlines()[-1:] or [""]
            try:
                if not json.loads(last[0])["replay_outcome_matches"]:
                    bad.append(f"{t}/{os.path.basename(lg)}: replay outcome differs")
            except Exception:
                bad.append(f"{t}/{os.path.basename(lg)}: render failed")
    S = lambda s, t: res[s].get(t, ("-", ""))
    with open(f"{d}/INDEX.tsv", "w") as f:
        f.write("seed\t" + "\t".join(TAGS) + "\t" + "\t".join(f"video_{t}" for t in TAGS) + "\n")
        for s in sorted(res):
            f.write(f"{s}\t" + "\t".join(S(s, t)[0] for t in TAGS) + "\t" + "\t".join(S(s, t)[1] for t in TAGS) + "\n")
    m, e = "mugM_nv_step3000", "mugE_nv_step3000"
    with open(f"{d}/M3000_vs_E3000.tsv", "w") as f:
        f.write("seed\tcase\tvideo_M3000\tvideo_E3000\n")
        for case, (om, oe) in (("E_only_success", ("FAIL", "SUCCESS")), ("M_only_success", ("SUCCESS", "FAIL"))):
            for s in sorted(res):
                if S(s, m)[0] == om and S(s, e)[0] == oe:
                    f.write(f"{s}\t{case}\t{S(s, m)[1]}\t{S(s, e)[1]}\n")
    cnt = {t: sum(S(s, t)[0] == "SUCCESS" for s in res) for t in TAGS}
    n = {t: sum(S(s, t)[0] != "-" for s in res) for t in TAGS}
    with open(f"{d}/README.txt", "w") as f:
        f.write("hanging_mug eval rollouts (seeds100), rendered by bit-exact action replay (p1_render.py).\n"
                "Frame: head | left wrist | right wrist camera; overlay: seed, final outcome, step, gripper values.\n\n")
        for t in TAGS:
            f.write(f"{t}: {cnt[t]}/{n[t]} success\n")
        f.write("\nINDEX.tsv: per-seed outcome and video for every checkpoint\n"
                "M3000_vs_E3000.tsv: seeds where Mix step 3000 and expert-only step 3000 disagree\n")
        f.write("\nproblems: " + ("none" if not bad else "\n" + "\n".join(bad)) + "\n")
    print(open(f"{d}/README.txt").read())


if __name__ == "__main__":
    main(sys.argv[1])
