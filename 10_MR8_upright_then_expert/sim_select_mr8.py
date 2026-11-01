"""Simulate-and-select driver for MR8 (upright-then-expert) (Claude, 2026-11-01). Same protocol as code/sim_select.py:
per failure state, 1 list feasible candidates (takeover_bench_mr8 --mr-list), 2 try them in order, each a fresh process
from the bit-exact replayed state, 3 keep the first attempt that passes every gate (quality_gates_mr8 G1-G6 plus the
joint gates G7/G8, always on) with a lossless capture. Crashes never produce a verdict.
usage: python sim_select_mr8.py <jobs.jsonl> <out_dir> <gpu,gpu,...> [max_candidates=6] [variant[,variant...]] [list_only=0]
Several comma-separated variants: candidates of the first are tried first (each candidate keeps its variant).
Job line: {"rollout", "step", "group", ...}. Writes <out_dir>/<group>_<seed>/{list.json, cand_*.json/.mp4/.npz/.log,
selection.json} and <out_dir>/summary_<hostname>.jsonl.
"""
import glob
import json
import os
import queue
import socket
import subprocess
import sys
import threading

import quality_gates_mr8 as QG8

HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.dirname(HERE)
VARIANT = "pp+seeds+rack+jboxz+ns+boxchk"
_ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


Z_MAX = 5.63  # round-1 audit (audit_mix_lists): reject state |z| above the expert maximum (5.13) + 0.5


def joint_gates(npz):
    """G7 (single-frame arm step <= 0.2 rad), Z (every recovery frame's arm state within |z| <= Z_MAX under the
    deployed FACT model's state normalization: the usability criterion for distillation) and G8 (asset range,
    expert + prefixes +-0.5 rad); all three required."""
    import numpy as np
    meta = json.load(open(os.path.join(L, "assets/rf_hanging_mug_v1/meta.json")))
    lo, hi = np.array(meta["G8_range"]["lo"]), np.array(meta["G8_range"]["hi"])
    ns = json.load(open(os.path.join(HERE, "fact_norm_stats_delta.json")))["norm_stats"]["observation.state"]
    sm, ss = np.array(ns["mean"]), np.array(ns["std"])
    q = np.load(npz)["expert_state"]
    step = float(np.abs(np.diff(q[:, _ARM], axis=0)).max()) if len(q) > 1 else 0.0
    z = float(np.abs((q - sm) / ss)[:, _ARM].max())
    # G8 required again (11-01 audit of v4: two kept left-arm z-box recoveries went up to 0.9 rad beyond the expert +
    # prefix range, |q| > pi on 101 frames, yet passed Z because FACT's std for those joints is wide)
    return {"G7_no_joint_snap": step <= 0.2, "Z_state": z <= Z_MAX, "z_state_max": round(z, 2), "max_step": round(step, 3),
            "G8_joint_range": not bool(((q < lo) | (q > hi))[:, _ARM].any())}


def required(g):
    return all(v for k, v in g.items() if k.startswith(("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "Z_", "L900")))


EARLY_STOP = 3  # v3: stop a state after this many consecutive attempts that never held the mug (grasp_fail)


def bench(gpu, roll, step, variant, out, extra):
    cmd = (f"source {HERE}/env_mr8.sh {gpu}; timeout 3000 python {HERE}/takeover_bench_mr8.py --rollout {roll} "
           f"--step {step} --plan MR8 --variant {variant} --out {out} {extra} > {out[:-5]}.log 2>&1")
    subprocess.run(["bash", "-c", cmd])
    return json.load(open(out)) if os.path.exists(out) else None


def process(job, gpu, out_dir, max_cands, variant, list_only):
    roll, step = job["rollout"], job["step"]
    seed = json.load(open(roll))["env_seed"]
    name = f'{job["group"]}_{seed}'
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    sel_path = os.path.join(d, "selection.json")
    if os.path.exists(sel_path):
        return json.load(open(sel_path))
    claim = os.path.join(d, ".claim")
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{socket.gethostname()} {os.getpid()}".encode())
        os.close(fd)
    except FileExistsError:
        return None
    cands = []
    for vi, v in enumerate(variant.split(",")):  # several planning variants (e.g. MR7 box first, then the z box)
        r = bench(gpu, roll, step, v, os.path.join(d, "list.json" if vi == 0 else f"list_v{vi}.json"), "--mr-list")
        if r is None:
            os.remove(claim)
            return {"name": name, "error": f"list v{vi} no output"}
        feas = next((e.get("feasible", []) for e in r.get("mr_log", []) if e["stage"] == "chain"), [])
        cands += [{**f, "variant": v} for f in feas]
        if any(f.get("direct") for f in feas):
            break  # already upright: the expert-only candidate needs no other planning variant
    tried, selected = [], None
    for i, c in enumerate([] if list_only else cands[:max_cands]):
        out = os.path.join(d, f"cand_{i:02d}.json")
        forced = {k: v for k, v in c.items() if k != "variant"}
        r = bench(gpu, roll, step, c["variant"], out, f"--video --capture --mr-forced '{json.dumps(forced)}'")
        if r is None:
            os.remove(claim)
            return {"name": name, "error": f"cand_{i:02d} no output"}
        g = QG8.gates(r)
        cap = r.get("capture") or {}
        npz = cap.get("npz")
        if npz and os.path.exists(npz):
            g.update(joint_gates(npz))
            # v3: the whole episode (policy prefix + recovery) must fit the 900-step evaluation limit, else a policy
            # that imitates it cannot finish in time
            g["L900"] = cap.get("prefix_frames", 0) + cap.get("expert_frames", 0) <= 900
        ok = bool(npz) and "Z_state" in g and "G8_joint_range" in g and required(g)
        res_path = next((e.get("result") for e in reversed(r.get("mr_log") or []) if e["stage"] == "path"), None)
        tried.append({**c, "idx": i, "status": r["status"], "layer": r["layer"], "gates": g, "pass": ok, "result": res_path})
        if ok:
            selected = {**c, "idx": i, "json": out, "video": out[:-5] + ".mp4"}
            break
        if len(tried) >= EARLY_STOP and all(t.get("result") == "grasp_fail" for t in tried[-EARLY_STOP:]):
            break
    res = {"name": name, "rollout": roll, "step": step, "job": job, "n_candidates": len(cands), "tried": tried,
           "selected": selected, "list_only": bool(list_only)}
    json.dump(res, open(sel_path, "w"))
    return res


def main():
    jobs = [json.loads(line) for line in open(sys.argv[1])]
    out_dir, gpus = sys.argv[2], sys.argv[3].split(",")
    max_cands = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    variant = sys.argv[5] if len(sys.argv) > 5 else VARIANT
    list_only = len(sys.argv) > 6 and sys.argv[6] == "1"
    os.makedirs(out_dir, exist_ok=True)
    import glob as _g
    for c in _g.glob(os.path.join(out_dir, "*", ".claim")):
        host, pid = (open(c).read().split() + ["", ""])[:2]
        if host == socket.gethostname() and not os.path.exists(os.path.join(os.path.dirname(c), "selection.json")) \
                and not os.path.exists(f"/proc/{pid}"):
            os.remove(c)
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    lock = threading.Lock()
    target = int(os.environ.get("SIM_SELECT_TARGET", "0"))  # 11-01: stop claiming new states once this many are kept
    kept = [sum(1 for p in glob.glob(os.path.join(out_dir, "*", "selection.json"))
                if (json.load(open(p)).get("selected")))]

    def worker(gpu):
        while True:
            if target and kept[0] >= target:
                return
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            r = process(j, gpu, out_dir, max_cands, variant, list_only)
            if r is None:
                continue
            if "error" in r:
                print(r["name"], "ERROR", r["error"], flush=True)
                continue
            with lock:
                kept[0] += bool(r["selected"])
                with open(os.path.join(out_dir, f"summary_{socket.gethostname()}.jsonl"), "a") as f:
                    f.write(json.dumps({"name": r["name"], "n_candidates": r["n_candidates"], "n_tried": len(r["tried"]),
                                        "selected": r["selected"]}) + "\n")
                print(r["name"], "candidates", r["n_candidates"], "tried", len(r["tried"]),
                      "SELECTED" if r["selected"] else "none", flush=True)

    ts = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("done")


if __name__ == "__main__":
    main()
