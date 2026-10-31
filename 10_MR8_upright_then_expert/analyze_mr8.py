"""Funnel / efficiency / usability summary of an MR8 run directory (Claude, 2026-11-01). CPU only, read-only.

usage: python analyze_mr8.py <run dir> <jobs.jsonl> [out.json]
Per MR6-failure bucket (job "cat"): states with a verdict, "direct" (already upright, expert only), states with >= 1
feasible restage candidate, by grasp family (rim / handle) and approach yaw; for executed runs: selected states, the
first failing stage of rejected attempts, gate rejections, the expert remainder used, upright quality, spot distance
and handle yaw of selected attempts (diversity), recovery length, max state |z|, G8 (info), wall time per state.
"""
import collections
import glob
import json
import os
import statistics as st
import sys


def load(p):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return None


def chain(d):
    return next((e for e in (d or {}).get("mr_log") or [] if e["stage"] == "chain"), {})


def main(run, jobs_path, out=None):
    jobs = {json.loads(x)["state_id"]: json.loads(x) for x in open(jobs_path)}
    by_seed = {int(j["state_id"].split("/")[1]): j for j in jobs.values()}
    rows = []
    for sp in sorted(glob.glob(os.path.join(run, "*", "selection.json"))):
        d = os.path.dirname(sp)
        S = load(sp)
        seed = int(os.path.basename(d).split("_")[-1])
        job = by_seed.get(seed, {})
        L = load(os.path.join(d, "list.json"))
        ch = chain(L)
        feas = ch.get("feasible", [])
        walls = [(load(p) or {}).get("wall_s", 0) for p in glob.glob(os.path.join(d, "*.json"))
                 if not p.endswith("selection.json")]
        row = {"seed": seed, "cat": job.get("cat", "?"), "tier": job.get("tier", "?"),
               "direct": bool(ch.get("direct")), "n_feasible": len(feas),
               "families": sorted({("handle" if isinstance(f.get("cid"), str) else "rim") for f in feas if "cid" in f}),
               "yaws": sorted({f.get("yaw") for f in feas if "yaw" in f}),
               "tried": len(S.get("tried", [])), "selected": bool(S.get("selected")), "wall_s": round(sum(walls), 1)}
        sel = S.get("selected")
        if sel:
            r = load(sel["json"]) or {}
            log = r.get("mr_log") or []
            up = next((e for e in log if e["stage"] == "uprighted"), {})
            hang = next((e for e in log if e["stage"] == "hang"), {})
            t = next(x for x in S["tried"] if x.get("pass"))
            row.update({"sel_family": "direct" if sel.get("direct") else ("handle" if isinstance(sel.get("cid"), str) else "rim"),
                        "sel_cid": sel.get("cid"), "sel_yaw": sel.get("yaw"), "spot": sel.get("spot"),
                        "alpha": sel.get("alpha"), "alpha_now": up.get("alpha_now"), "upright_deg": up.get("upright_deg"),
                        "spot_err_m": up.get("spot_err_m"), "expert": hang.get("expert"),
                        "frames": (r.get("capture") or {}).get("expert_frames"),
                        "prefix_frames": (r.get("capture") or {}).get("prefix_frames"),
                        "z_state_max": t["gates"].get("z_state_max"), "max_step": t["gates"].get("max_step"),
                        "G8_info": t["gates"].get("G8_joint_range_info")})
        fails = collections.Counter()
        for x in S.get("tried", []):
            if x.get("pass"):
                continue
            g = x.get("gates", {})
            bad = [k.split("_")[0] for k, v in g.items() if v is False and not k.endswith("_info")]
            r = load(os.path.join(d, f"cand_{x['idx']:02d}.json")) or {}
            res = next((e.get("result") for e in reversed(r.get("mr_log") or []) if e["stage"] == "path"), None)
            fails[f"{res or r.get('status')}|{','.join(bad)}"] += 1
        row["reject"] = dict(fails)
        rows.append(row)
    cats = collections.defaultdict(list)
    for r in rows:
        cats[r["cat"]].append(r)
    summ = {}
    for c in sorted(cats) + ["ALL"]:
        R = rows if c == "ALL" else cats[c]
        n = len(R)
        if not n:
            continue
        sel = [r for r in R if r["selected"]]
        summ[c] = {"states": n, "direct": sum(r["direct"] for r in R),
                   "with_candidate": sum(1 for r in R if r["direct"] or r["n_feasible"] > 0),
                   "with_handle_cand": sum(1 for r in R if "handle" in r["families"]),
                   "with_rim_cand": sum(1 for r in R if "rim" in r["families"]),
                   "selected": len(sel), "yield": round(len(sel) / n, 3),
                   "attempts": sum(r["tried"] for r in R),
                   "wall_h": round(sum(r["wall_s"] for r in R) / 3600, 2),
                   "sel_family": dict(collections.Counter(r.get("sel_family") for r in sel)),
                   "expert": dict(collections.Counter(r.get("expert") for r in sel)),
                   "frames_median": st.median([r["frames"] for r in sel if r.get("frames")]) if sel else None,
                   "total_le_900": sum(1 for r in sel if r.get("frames") and r["frames"] + (r.get("prefix_frames") or 0) <= 900),
                   "distinct_spots": len({tuple(r["spot"]) for r in sel if r.get("spot")}),
                   "alphas": dict(collections.Counter(r.get("alpha") for r in sel if r.get("alpha") is not None)),
                   "z_max": max([r["z_state_max"] for r in sel if r.get("z_state_max") is not None], default=None),
                   "G8_info_pass": sum(1 for r in sel if r.get("G8_info")),
                   "rejects": dict(sum((collections.Counter(r["reject"]) for r in R), collections.Counter()))}
    res = {"run": run, "summary": summ, "rows": rows}
    print(json.dumps(summ, indent=1))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
