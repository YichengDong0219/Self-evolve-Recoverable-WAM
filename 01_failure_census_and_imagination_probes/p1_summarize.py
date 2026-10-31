"""P1 per-episode fact summary from dense labels (Claude, 2026-10-27). CPU only, exploratory (development set).

Per episode: grasp closures per side, contact segments per side, lift episodes, last object contact, terminal
stall window (no target contact and object/joint motion below threshold until the end), final object state.
Side mapping: robot links 'fl_*' = left arm, 'fr_*' = right arm (aloha-agilex naming in these scenes).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

MOVE_EPS = 0.005  # m (object) / normalized joint fraction (switch) treated as "no motion"


def side_of(link):
    return "L" if link.startswith("fl_") else "R" if link.startswith("fr_") else None


def segments(flags):
    segs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if not f and start is not None:
            segs.append((start, i - 1)); start = None
    if start is not None:
        segs.append((start, len(flags) - 1))
    return segs


def summarize(path):
    d = json.loads(Path(path).read_text())
    rows, task = d["rows"], d["task"]
    contact = {s: [any(side_of(l) == s for l in r["contacts"]) for r in rows] for s in "LR"}
    closed = {s: [r["grip"][0 if s == "L" else 1] < 0.2 for r in rows] for s in "LR"}
    out = {"episode": Path(path).name.replace(".label.json", ""), "task": task, "env_seed": d["env_seed"],
           "success": d["success"], "steps": d["steps"], "replay_ok": d["replay_outcome_matches"]}
    for s in "LR":
        out[f"close_events_{s}"] = len(segments(closed[s]))
        out[f"contact_segs_{s}"] = segments(contact[s])[:8]
    anyc = [contact["L"][i] or contact["R"][i] for i in range(len(rows))]
    last_contact = max([i for i, c in enumerate(anyc) if c], default=None)
    out["last_contact_step"] = rows[last_contact]["step"] if last_contact is not None else None
    if task == "turn_switch":
        lo, hi = rows[0]["q_lim"]
        frac = [(r["q"] - lo) / (hi - lo) for r in rows]
        series = np.array(frac)[:, None]
        out["q_frac_max"] = round(float(max(frac)), 3)
        out["q_frac_final"] = round(float(frac[-1]), 3)
    else:
        series = np.array([r["p"] for r in rows])
        z0 = series[0, 2]
        lifted = [r["p"][2] > z0 + 0.03 for r in rows]
        out["lift_segs"] = segments(lifted)[:6]
        out["final"] = {"p": rows[-1]["p"], "tilt_deg": rows[-1]["tilt_deg"], "tilt0_deg": rows[0]["tilt_deg"],
                        "dxy": rows[-1]["dxy"], "dz": rows[-1]["dz"], "on_table": bool(rows[-1]["p"][2] < z0 + 0.01)}
        out["min_dxy"] = round(min(r["dxy"] for r in rows), 4)
    # terminal stall: largest suffix with no target contact and object displacement < MOVE_EPS from final
    final = series[-1]
    k = len(rows) - 1
    while k > 0 and not anyc[k - 1] and np.linalg.norm(series[k - 1] - final) < MOVE_EPS:
        k -= 1
    out["terminal_stall_from"] = rows[k]["step"]
    out["terminal_stall_len"] = rows[-1]["step"] - rows[k]["step"]
    return out


if __name__ == "__main__":
    for p in sorted(glob.glob(sys.argv[1] + "/*.label.json")):
        print(json.dumps(summarize(p)))
