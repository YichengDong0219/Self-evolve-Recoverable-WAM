"""Runtime instrumentation of the RoboTwin expert for takeover diagnosis (Claude, 2026-10-28). RoboTwin is not edited.

install(env, target) wraps, on this env instance only:
  - every action builder (grasp_actor, place_actor, move_by_displacement, back_to_origin, close/open_gripper,
    move_to_pose): the returned Actions are tagged with the primitive name (and contact_point_id if given);
  - both CuroboPlanner.motion_gen.plan_single / motion_gen_batch.plan_batch: the raw curobo status is kept
    (RoboTwin itself only reports "Success"/"Fail");
  - env.move: one record per call with the primitives, planner calls, plan_success before/after, target-object
    pose / holder before and after, and physical post-condition flags.
classify(trace, ok) turns the trace into one failure layer:
  success | manipulation:no_grasp_pose | motion:<primitive>:<curobo status> | physical:<flag> | goal_not_met
"""
import functools

import numpy as np

BUILDERS = ["grasp_actor", "place_actor", "move_by_displacement", "back_to_origin", "close_gripper", "open_gripper",
            "move_to_pose"]


def holder(env, obj):
    name = obj.get_name()
    links = set()
    for c in env.scene.get_contacts():
        n0, n1 = c.bodies[0].entity.name, c.bodies[1].entity.name
        if name in (n0, n1):
            links.add(n1 if n0 == name else n0)
    if any(x.startswith("fl_") for x in links):
        return "left"
    if any(x.startswith("fr_") for x in links):
        return "right"
    return "none"


def obj_state(env, obj):
    p = obj.get_pose()
    return {"p": np.round(p.p, 4).tolist(), "q": np.round(p.q, 4).tolist(), "holder": holder(env, obj)}


def install(env, target):
    trace = {"moves": [], "plans": [], "builder_errors": []}
    current = {"prims": []}

    def wrap_builder(name):
        orig = getattr(env, name)

        @functools.wraps(orig)
        def f(*args, **kwargs):
            try:
                out = orig(*args, **kwargs)
            except Exception as e:  # noqa: BLE001  e.g. Action(..., target_pose=None) when no grasp pose exists
                trace["builder_errors"].append({"prim": name, "cid": kwargs.get("contact_point_id"),
                                                "error": f"{type(e).__name__}: {e}"})
                raise
            try:
                arm, acts = out
                for a in acts or []:
                    a.args["_prim"] = name
                    if "contact_point_id" in kwargs:
                        a.args["_cid"] = kwargs["contact_point_id"]
            except (TypeError, ValueError, AttributeError):
                pass
            return out
        setattr(env, name, f)

    for b in BUILDERS:
        wrap_builder(b)

    def wrap_plan(planner, arm, attr, method):
        mg = getattr(planner, attr)
        orig = getattr(mg, method)

        @functools.wraps(orig)
        def f(*args, **kwargs):
            res = orig(*args, **kwargs)
            try:
                succ = res.success.cpu().numpy().reshape(-1).tolist()
            except Exception:  # noqa: BLE001
                succ = None
            trace["plans"].append({"arm": arm, "kind": method, "prims": list(current["prims"]),
                                   "status": str(getattr(res, "status", None)), "success": succ,
                                   "attempts": int(getattr(res, "attempts", -1) or -1),
                                   "valid_query": bool(getattr(res, "valid_query", True))})
            return res
        setattr(mg, method, f)

    for arm, pl in (("left", env.robot.left_planner), ("right", env.robot.right_planner)):
        wrap_plan(pl, arm, "motion_gen", "plan_single")
        wrap_plan(pl, arm, "motion_gen_batch", "plan_batch")

    orig_move = env.move

    @functools.wraps(orig_move)
    def move(*args, **kwargs):
        acts = [a for x in args if isinstance(x, tuple) and x and x[1] for a in x[1]]
        prims = [{"prim": a.args.get("_prim", "?"), "arm": str(a.arm_tag), "action": a.action,
                  "cid": a.args.get("_cid"), "target_none": a.action == "move" and a.target_pose is None}
                 for a in acts if a is not None]
        current["prims"] = [p["prim"] for p in prims]
        before = obj_state(env, target)
        ps_before = bool(env.plan_success)
        n_plans = len(trace["plans"])
        q0 = np.asarray(env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState(), dtype=float)
        err = None
        try:
            ret = orig_move(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            trace["moves"].append({"prims": prims, "error": err, "plan_success_before": ps_before,
                                   "plan_success_after": False, "before": before, "after": obj_state(env, target),
                                   "plans": list(range(n_plans, len(trace["plans"])))})
            raise
        after = obj_state(env, target)
        q1 = np.asarray(env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState(), dtype=float)
        disp = float(np.linalg.norm(np.asarray(after["p"]) - np.asarray(before["p"])))
        flags = []
        kinds = {p["action"] for p in prims}
        if before["holder"] == "none" and after["holder"] == "none" and disp > 0.02:
            flags.append("object_disturbed")
        dz = float(before["p"][2] - after["p"][2])
        # "dropped" needs a real fall: a mug merely resting against an open gripper loses that contact when the arm
        # moves away (A6 capture test: 0.5 mm motion was flagged and failed G6)
        if before["holder"] != "none" and after["holder"] == "none" and "gripper" not in kinds and dz > 0.02:
            flags.append("object_dropped")
        if after["p"][2] < 0.6:
            flags.append("object_off_table")
        if disp > 0.5:
            flags.append("object_flung")
        trace["moves"].append({"prims": prims, "error": None, "plan_success_before": ps_before,
                               "plan_success_after": bool(env.plan_success), "before": before, "after": after,
                               "obj_disp": round(disp, 4), "joint_disp": round(float(np.abs(q1 - q0).max()), 3),
                               "flags": flags, "plans": list(range(n_plans, len(trace["plans"])))})
        return ret
    env.move = move
    return trace


def classify(trace, ok, error=None):
    if ok:
        return "success"
    for b in trace["builder_errors"]:
        if "target_pose cannot be None" in b["error"]:
            return f"manipulation:no_grasp_pose:{b['prim']}"
    for m in trace["moves"]:
        if any(p["target_none"] for p in m["prims"]) or (m["error"] and "target_pose cannot be None" in m["error"]):
            return "manipulation:no_grasp_pose"
        if m["plan_success_before"] and not m["plan_success_after"]:
            failed = [trace["plans"][i] for i in m["plans"] if trace["plans"][i]["success"] is not None
                      and not all(trace["plans"][i]["success"])]
            st = failed[-1]["status"].split(".")[-1] if failed else "no_curobo_call"
            prim = (failed[-1]["prims"] or ["?"])[0] if failed else (m["prims"][0]["prim"] if m["prims"] else "?")
            return f"motion:{prim}:{st}"
    for m in trace["moves"]:
        if m.get("flags"):
            return f"physical:{m['flags'][0]}"
    if error:
        return f"error:{error.split(':')[0]}"
    return "goal_not_met"
