"""MR7 smoke-test report (Claude, 2026-10-30): efficiency and action quality of the MR7 takeover vs MR6 on the same
asset states. Reads bench/SEL_mr7_smoke (copied from Amax), the asset index (MR6 evidence) and FACT's norm stats.
usage: python mr7_smoke_report.py <sel_dir> <jobs.jsonl> <norm_stats_delta.json> [out.json]
"""
import glob
import json
import os
import sys

import numpy as np

L = "/workspace/fact_local"
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
G7_MAX = 0.2


def fix(p):
    """results produced on the cluster carry its paths; they were copied into the same tree here"""
    return p.replace("/workspace_cluster/fact_local", L) if p and p.startswith("/home/") else p


def main(sel_dir, jobs, ns_path, out=None):
    idx = {json.loads(x)["state_id"]: json.loads(x) for x in open(f"{L}/assets/rf_hanging_mug_v1/index.jsonl")}
    meta = json.load(open(f"{L}/assets/rf_hanging_mug_v1/meta.json"))
    g8lo, g8hi = np.array(meta["G8_range"]["lo"]), np.array(meta["G8_range"]["hi"])
    box = json.load(open(f"{L}/code/mr7_joint_box.json"))
    blo = np.array(box["left"]["lo"] + [-9] + box["right"]["lo"] + [-9])
    bhi = np.array(box["left"]["hi"] + [9] + box["right"]["hi"] + [9])
    ns = json.load(open(ns_path))["norm_stats"]
    sm, ss = np.array(ns["observation.state"]["mean"]), np.array(ns["observation.state"]["std"])
    E = np.concatenate([np.load(json.loads(x)["npz"])["expert_state"]
                        for x in open(f"{L}/sft_lists_final_g7_1029/E_expert.jsonl")])
    wr_lo, wr_hi = E[:, [5, 12]].min(0), E[:, [5, 12]].max(0)  # wrist roll range of the expert demos
    rows = []
    for j in map(json.loads, open(jobs)):
        sid, cat = j["state_id"], j["smoke_cat"]
        seed = int(sid.split("/")[1])
        d = f"{sel_dir}/{j['group']}_{seed}"
        r = {"state_id": sid, "cat": cat}
        mr6 = idx[sid]["mr6"]
        r["mr6"] = {"tier": idx[sid]["mr6_tier"], "n_cand": mr6["n_candidates"], "n_att": mr6["n_attempts"],
                    "G8": (mr6["selected"] or {}).get("G8_pass")}
        if not os.path.exists(f"{d}/selection.json"):
            r["mr7"] = None
            rows.append(r)
            continue
        s = json.load(open(f"{d}/selection.json"))
        walls, boxchk, checked = [], 0, 0
        for f in glob.glob(f"{d}/list_v*.json") + glob.glob(f"{d}/cand_*.json"):
            b = json.load(open(f))
            walls.append(b.get("wall_s") or 0)
            boxchk += sum(e.get("fix") == "boxchk" for e in b.get("fix_log", []))
            checked += (b.get("box_stats") or {}).get("checked", 0)
        m7 = {"n_cand": s.get("n_candidates", 0), "n_flip_cand": None, "n_att": len(s.get("tried", [])),
              "recovered": bool(s.get("selected")), "wall_s": round(sum(walls), 1), "boxchk_rejected": boxchk,
              "plans_checked": checked, "error": s.get("error")}
        lists = [json.load(open(f)) for f in glob.glob(f"{d}/list_v*.json")]
        feas = [f for lj in lists for e in lj.get("mr_log", []) if e.get("stage") == "chain" for f in e.get("feasible", [])]
        m7["n_flip_cand"] = sum(bool(f.get("flip")) for f in feas)
        m7["attempt_gates"] = [{k: v for k, v in t["gates"].items() if not v} for t in s.get("tried", [])]
        if s.get("selected"):
            sel = s["selected"]
            sel = {**sel, "json": fix(sel["json"]), "video": fix(sel["video"])}
            cj = json.load(open(sel["json"]))
            npz = fix((cj.get("capture") or {}).get("npz"))
            m7["selected"] = {"variant": sel["variant"], "arm": sel["arm"], "cid": sel["cid"],
                              "flip": bool(sel.get("flip", False)), "video": sel["video"], "npz": npz}
            if npz and os.path.exists(npz):
                q = np.load(npz)["expert_state"]
                step = np.abs(np.diff(q[:, ARM], axis=0)).max()
                oor_box = ((q < blo - 1e-3) | (q > bhi + 1e-3))[:, ARM].any(1).sum()
                oor_g8 = ((q < g8lo) | (q > g8hi))[:, ARM].any(1).sum()
                z = np.abs((q - sm) / ss)[:, ARM].max()
                wr = q[:, [5, 12]]
                m7["selected"].update(frames=int(len(q)), max_joint_step=round(float(step), 4),
                                      G7=bool(step <= G7_MAX), frames_out_of_box=int(oor_box),
                                      G8=bool(oor_g8 == 0), max_abs_z_state=round(float(z), 2),
                                      wrist_in_expert_range=bool(((wr >= wr_lo - 0.05) & (wr <= wr_hi + 0.05)).all()),
                                      joint_min=np.round(q[:, ARM].min(0), 2).tolist(),
                                      joint_max=np.round(q[:, ARM].max(0), 2).tolist())
        r["mr7"] = m7
        rows.append(r)
    done = [r for r in rows if r["mr7"] and not r["mr7"].get("error")]
    summ = {"states": len(rows), "finished": len(done)}
    for cat in sorted({r["cat"] for r in rows}):
        g = [r for r in done if r["cat"] == cat]
        summ[cat] = {"n": len(g), "mr7_feasible": sum(r["mr7"]["n_cand"] > 0 for r in g),
                     "mr7_recovered": sum(r["mr7"]["recovered"] for r in g),
                     "mr6_recovered": sum(r["mr6"]["tier"] == "A_recovered" for r in g),
                     "mr6_recovered_G8": sum(bool(r["mr6"]["G8"]) for r in g)}
    sel = [r["mr7"]["selected"] for r in done if r["mr7"]["recovered"] and "frames" in r["mr7"]["selected"]]
    summ["selected_quality"] = {"n": len(sel), "G7": sum(x["G7"] for x in sel), "G8": sum(x["G8"] for x in sel),
                                "wrist_in_expert_range": sum(x["wrist_in_expert_range"] for x in sel),
                                "flip": sum(x["flip"] for x in sel),
                                "max_abs_z_state_max": max([x["max_abs_z_state"] for x in sel], default=None)}
    summ["boxchk_rejected_total"] = sum(r["mr7"]["boxchk_rejected"] for r in done)
    summ["plans_checked_total"] = sum(r["mr7"]["plans_checked"] for r in done)
    summ["wall_h_total"] = round(sum(r["mr7"]["wall_s"] for r in done) / 3600, 2)
    summ["wall_h_per_recovery"] = round(summ["wall_h_total"] / max(1, sum(r["mr7"]["recovered"] for r in done)), 2)
    print(json.dumps(summ, indent=1))
    if out:
        json.dump({"summary": summ, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
