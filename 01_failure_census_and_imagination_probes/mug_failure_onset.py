"""hanging_mug failure-onset detector from benchmark signals (Claude, 2026-10-27). CPU only, online-capable.

Input: a dense label file from p1_label.py (per step: mug pose p, tcp_l, tcp_r, ...). Rules (first to fire wins):
  a  decoupled fall : within 10 steps mug height drops >= 3 cm AND mug-to-nearest-TCP distance grows >= 3 cm
  b  back to table  : mug reached hanging height (z0 + 10 cm) and later returns below z0 + 2 cm
                      (onset = last step at hanging height before the return)
  c  stall, no lift : at step >= 150 the mug has never reached hanging height and nothing moved for 96 steps
                      (onset = start of the still window)
Development set: the 20 P1 hanging_mug demo_clean episodes -> 14/14 failures get an onset, 0/6 successes flagged.
Thresholds were chosen on that set; validate on held-out episodes before treating them as fixed.
"""
import json
import math
import sys


def onset(rows):
    z = [r["p"][2] for r in rows]
    z0 = z[0]
    sep = [min(math.dist(r["p"], r["tcp_l"]), math.dist(r["p"], r["tcp_r"])) for r in rows]
    hits = {}
    for i in range(len(rows) - 10):
        if z[i] - z[i + 10] >= 0.03 and sep[i + 10] - sep[i] >= 0.03:
            hits["a_decoupled_fall"] = i
            break
    hi = next((i for i, v in enumerate(z) if v > z0 + 0.10), None)
    if hi is not None:
        back = next((i for i in range(hi, len(rows)) if z[i] < z0 + 0.02), None)
        if back is not None:
            hits["b_back_to_table"] = max(i for i in range(hi, back) if z[i] > z0 + 0.10)
    still = 0
    for i in range(1, len(rows)):
        still = still + 1 if math.dist(rows[i]["p"], rows[i - 1]["p"]) < 1e-4 else 0
        if still >= 96 and (hi is None or hi > i) and rows[i]["step"] >= 150:
            hits["c_stall_no_lift"] = i - 96
            break
    if not hits:
        return None
    kind, idx = min(hits.items(), key=lambda kv: kv[1])
    return {"t_event": rows[idx]["step"], "rule": kind, "all_rules": {k: rows[v]["step"] for k, v in hits.items()}}


if __name__ == "__main__":
    for path in sys.argv[1:]:
        d = json.load(open(path))
        print(json.dumps({"label": path, "success": d["success"], "steps": d["steps"], "onset": onset(d["rows"])}))
