"""Compare two exec_candidates outputs of the same episode candidate by candidate (keyframes, end capture, outcome).
usage: python compare_exec.py <ref prefix> <test prefix>   (prefix = path without .json/.npz)"""
import json
import sys

import numpy as np



def flat(cap):   # copied verbatim from exec_candidates.py (which imports the simulator)
    vals = []
    for k in sorted(cap["actors"]):
        r = cap["actors"][k]
        vals += r["p"] + r["q"] + r.get("v", []) + r.get("w", [])
    for k in sorted(cap["articulations"]):
        r = cap["articulations"][k]
        vals += r["root_p"] + r["root_q"] + r["qpos"] + r["qvel"]
        for t in r["drive_target"] + r["drive_velocity_target"]:
            vals += t
    return np.asarray(vals, dtype=np.float64), (cap["np_rng_sha"], cap["py_rng_sha"], cap["action_step"])


def same(a, b):
    va, ka = flat(a)
    vb, kb = flat(b)
    if va.shape != vb.shape or ka != kb:
        return False, float("inf")
    d = float(np.max(np.abs(va - vb))) if va.size else 0.0
    return d == 0.0, d

ref, test = sys.argv[1], sys.argv[2]
rj, tj = json.load(open(ref + ".json")), json.load(open(test + ".json"))
rz, tz = np.load(ref + ".npz"), np.load(test + ".npz")
rrows = {(r["decision"], r["cand"]): r for r in rj["rows"]}
ok_all, n = True, 0
for r in tj["rows"]:
    k = (r["decision"], r["cand"]); q = rrows[k]; key = f"d{k[0]}_c{k[1]}"
    img_eq = bool(np.array_equal(rz[key], tz[key])) if key in rz.files and key in tz.files else (key not in rz.files and key not in tz.files)
    cap_eq = same(q["end_capture"], r["end_capture"])[0] if "end_capture" in q and "end_capture" in r else None
    out_eq = (q.get("success_within_chunk"), q.get("finished_at")) == (r.get("success_within_chunk"), r.get("finished_at"))
    ok = img_eq and cap_eq and out_eq and r["start_ok"]
    ok_all &= bool(ok); n += 1
    print(key, "start_ok", r["start_ok"], "images_equal", img_eq, "end_capture_equal", cap_eq, "outcome_equal", out_eq,
          "wall ref/test", q["wall_s"], r["wall_s"])
print("ALL_IDENTICAL" if ok_all else "MISMATCH", n, "candidates")
