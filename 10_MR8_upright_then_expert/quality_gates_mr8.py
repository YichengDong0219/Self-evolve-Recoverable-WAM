"""Quality gates for MR8 (upright-then-expert) attempts (Claude, 2026-11-01).

MR8 restages by design and ends with the stock expert, so two MR6 gates change meaning; the rest are kept:
  G1 success_stable     unchanged: success, and still successful 100 sim steps after the final release
  G2 no_homing_before   no back_to_origin BEFORE the stock expert starts (its first grasp_actor move); the expert's
                        own back_to_origin (left arm, in parallel with the right grasp) is the demonstration's style
  G3 single_restage     at most one restage grasp (no retries of the upright step)
  G4 upright_ok         the mug stood within 15 deg of upright before the expert started (or was upright already)
  G5 undisturbed        < 5 mm mug motion before the first close, counted from the last deliberate push (v4, as MR6)
  G6 no_physical_flag   no drop / off-table / fling / disturbed flag before the final hang move (push moves may disturb)
The pair-level joint gates G7 (single-frame step <= 0.2 rad) and G8 (asset joint range) are added by the driver.
"""
import json
import sys


def gates(d):
    moves = d["trace"]["moves"]
    log = d.get("mr_log") or []
    prims = [[p["prim"] for p in m["prims"]] for m in moves]
    expert0 = next((i for i, ps in enumerate(prims) if "grasp_actor" in ps), len(moves))
    closes = [i for i, m in enumerate(moves) if any(p["action"] == "gripper" and p["prim"] in ("grasp_actor", "?")
                                                    for p in m["prims"])]
    # v4: deliberate pushes (moves tagged "push", e.g. the mouth-hook drag) are exempt, as in MR6's quality_gates: G5
    # counts from the last push on, G6 ignores "object_disturbed" on push moves (drop / off-table / fling still count)
    pushes = [i for i, ps in enumerate(prims) if "push" in ps]
    last_push = pushes[-1] if pushes else -1
    first_close = next((i for i in closes if i > last_push), len(moves))
    hang_idx = max([i for i, ps in enumerate(prims) if "place_actor" in ps], default=len(moves))
    hang = next((e for e in log if e["stage"] == "hang"), {})
    before = next((e for e in log if e["stage"] == "before_expert"), {})
    return {
        "G1_success_stable": d["status"] == "SUCCESS" and bool(hang.get("success_after_100_steps", False)),
        "G2_no_homing_before_expert": not any("back_to_origin" in ps for ps in prims[:expert0]),
        "G3_single_restage": sum(1 for e in log if e["stage"] == "grasp+lift") <= 1,
        "G4_upright_ok": before.get("class") == "upright" and before.get("upright_deg", 99) <= 15,
        "G5_undisturbed": sum(m.get("obj_disp", 0) for m in moves[last_push + 1:first_close]) < 0.005,
        "G6_no_physical_flag": not any([f for f in m.get("flags", []) if not (i in pushes and f == "object_disturbed")]
                                       for i, m in enumerate(moves[:hang_idx])),
    }


if __name__ == "__main__":
    for p in sys.argv[1:]:
        d = json.load(open(p))
        g = gates(d)
        print(p, d["status"], "PASS" if all(g.values()) else [k for k, v in g.items() if not v])
