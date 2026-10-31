"""Line 2 (self-evolution data, Claude 2026-10-31): classify policy episodes and pick branch points. CPU only.

usage: python classify_selfevo.py <episode dir> <out.json>
Reads fact-g0-episode/1 JSONs (g0_driver rollout or branch). Per episode:
  outcome   one_shot  success, <= 400 steps and exactly one right-gripper close
            recovered success with > 400 steps or >= 2 right closes (the agreed recovery-success definition)
            fail
  phase     (failures) hang = the right arm lifted the mug and it fell after the first right release;
            handover = the right arm never lifted it; other. Only hang failures get branch points (user, 10-31).
  branch    48-step boundaries where the mug lies at rest near table height with the right gripper open, after a
            right release: the first one (first fall) plus the first one after each later release (post-failed-grasp),
            at most 1 + MAX_EXTRA per episode, and only if >= MIN_LEFT policy steps remain before the 900-step limit.
Also writes, for recovered successes, the boundary after the last fall (start of the final recovery) for masking.
"""
import glob
import json
import os
import sys

STEP_LIMIT, PLAN, LONG = 900, 48, 400
MAX_EXTRA, MIN_LEFT = 2, 200
LOW, REST_V = 0.05, 0.01  # mug within 5 cm of its initial height; speed below 1 cm/s counts as at rest


def grip_events(acts):
    ev = []
    for g, n in ((6, "L"), (13, "R")):
        st = acts[0][g] > 0.5
        for k in range(1, len(acts)):
            v = acts[k][g]
            ns = True if v > 0.5 else (False if v < 0.3 else st)
            if ns != st:
                ev.append((k, n + ("o" if ns else "c")))
                st = ns
    return sorted(ev)


def mug_track(ep):
    rows = []
    for k, v in ep["states"].items():
        if k == "final" or "actors" not in v:
            continue
        m = v["actors"]["0:039_mug"]
        rows.append((int(k), m["p"][2], sum(x * x for x in m.get("v", [0, 0, 0])) ** 0.5))
    return sorted(rows)


def right_open_at(acts, t):
    return acts[min(t, len(acts) - 1)][13] > 0.5


def classify(ep):
    acts, ev = ep["executed_actions"], grip_events(ep["executed_actions"])
    track = mug_track(ep)
    z0 = track[0][1]
    rcs = [k for k, e in ev if e == "Rc"]
    ros = [k for k, e in ev if e == "Ro"]
    out = {"env_seed": ep["env_seed"], "steps": ep["steps"], "success": ep["success"], "right_closes": len(rcs),
           "mode": ep["mode"], "branch_step": ep.get("branch_step", 0), "root_seed": ep.get("root_seed")}
    rest = [t for t, z, v in track if z < z0 + LOW and v < REST_V]
    falls = []  # first rest boundary after each right release
    for r in ros:
        b = next((t for t in rest if t > r and right_open_at(acts, t)), None)
        if b is not None and b not in falls:
            falls.append(b)
    if ep["success"]:
        out["outcome"] = "one_shot" if ep["steps"] <= LONG and len(rcs) <= 1 else "recovered"
        if out["outcome"] == "recovered":
            done = [b for b in falls if b < ep["steps"]]
            out["last_fall_boundary"] = done[-1] if done else None
        return out
    out["outcome"] = "fail"
    if not rcs:
        out["phase"] = "other"
    else:
        after = [z for t, z, _ in track if t >= rcs[0] + PLAN]
        if not after or max(after) < z0 + LOW:
            out["phase"] = "handover"
        else:
            peak_t = max((z, t) for t, z, _ in track if t >= rcs[0])[1]
            fall_t = next((t for t, z, _ in track if t > peak_t and z < z0 + LOW), None)
            out["phase"] = "hang" if fall_t is not None and ros and fall_t >= ros[0] else "other"
    if out["phase"] == "hang":
        pts = [b for b in falls if b <= STEP_LIMIT - MIN_LEFT]
        out["branch_points"] = [{"step": b, "kind": "first_fall" if i == 0 else "post_failed_grasp"}
                                for i, b in enumerate(pts[:1 + MAX_EXTRA])]
    return out


def main(d, out):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            ep = json.load(open(p))
        except json.JSONDecodeError:
            continue
        if ep.get("format") != "fact-g0-episode/1":
            continue
        r = classify(ep)
        r["file"] = p
        rows.append(r)
    summ = {"episodes": len(rows)}
    for k in ("one_shot", "recovered", "fail"):
        summ[k] = sum(r["outcome"] == k for r in rows)
    summ["fail_phase"] = {ph: sum(r.get("phase") == ph for r in rows) for ph in ("hang", "handover", "other")}
    summ["branch_points"] = {k: sum(b["kind"] == k for r in rows for b in r.get("branch_points", []))
                             for k in ("first_fall", "post_failed_grasp")}
    json.dump({"summary": summ, "rows": rows}, open(out, "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main(*sys.argv[1:])
