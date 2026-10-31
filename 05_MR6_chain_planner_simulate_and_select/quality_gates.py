"""SFT data-quality gates for hanging_mug recovery attempts (Claude, 2026-10-28).

usage: python quality_gates.py <bench variant dir> [...]
A recovery enters SFT only if it passes every gate (the user's priority: quality over yield):
  G1 success_stable   task success, and for MR6 still successful 100 sim steps after release
  G2 no_homing        no back_to_origin anywhere in the recovery
  G3 single_attempt   exactly one gripper close on the mug (no retries)
  G4 direct           pick where it fell and hang; no restage / hand-over / upright placement
  G5 undisturbed      the mug moved < 5 mm in total before the grasp closed (no shoving on the way); deliberate
                      reorientation pushes (moves tagged "push") are excluded: counted from the last push on
  G6 no_physical_flag no drop / off-table / fling flagged by takeover_diag on any move BEFORE the hang move (after the
                      gripper opens on the hook the mug settling against the rod is expected; G1 covers the end state)
Prints per-attempt verdicts and a summary per variant.
"""
import glob
import json
import os
import sys


def gates(d):
    moves = d["trace"]["moves"]
    prims = [p for m in moves for p in m["prims"]]
    log = d.get("mr_log") or []
    stages = {e["stage"] for e in log}
    closes = [i for i, m in enumerate(moves) if any(p["action"] == "gripper" and p["prim"] in ("grasp_actor", "?")
                                                    and not str(p.get("prim")).startswith("open") for p in m["prims"])]
    pushes = [i for i, m in enumerate(moves) if any(p["prim"] == "push" for p in m["prims"])]
    last_push = pushes[-1] if pushes else -1
    first_close = next((i for i in closes if i > last_push), len(moves)) if closes else len(moves)
    hang_idx = next((i for i, m in enumerate(moves) if any(p["prim"] == "place_actor" for p in m["prims"])), len(moves))
    hang = next((e for e in log if e["stage"] == "hang"), {})
    stable = hang.get("success_after_100_steps", d["status"] == "SUCCESS") if hang else d["status"] == "SUCCESS"
    g = {
        "G1_success_stable": d["status"] == "SUCCESS" and bool(stable),
        "G2_no_homing": not any(p["prim"] == "back_to_origin" for p in prims),
        "G3_single_attempt": sum(1 for e in log if e["stage"] == "grasp+lift") == 1,
        "G4_direct": not ({"restage", "restaged", "handover_place", "handed_over", "spot"} & stages),
        "G5_undisturbed": sum(m.get("obj_disp", 0) for m in moves[last_push + 1:first_close]) < 0.005,
        "G6_no_physical_flag": not any(m.get("flags") for m in moves[:hang_idx]),
    }
    return g


def main():
    for vdir in sys.argv[1:]:
        rows = []
        for p in sorted(glob.glob(os.path.join(vdir, "*.json"))):
            d = json.load(open(p))
            g = gates(d)
            rows.append((os.path.basename(p)[:-5], d["status"], g))
        print(f"=== {os.path.basename(vdir.rstrip('/'))}: {len(rows)} attempts")
        for name, st, g in rows:
            bad = [k for k, v in g.items() if not v]
            print(f"  {name[:28]:28s} {st:8s} {'PASS' if not bad else 'reject: ' + ','.join(b.split('_', 1)[0] for b in bad)}")
        n_ok = sum(1 for _, st, _ in rows if st == "SUCCESS")
        n_pass = sum(1 for _, _, g in rows if all(g.values()))
        per = {k: sum(1 for _, st, g in rows if st == "SUCCESS" and not g[k]) for k in rows[0][2]} if rows else {}
        print(f"  successes {n_ok}, pass all gates {n_pass}; successes rejected per gate: {per}")


if __name__ == "__main__":
    main()
