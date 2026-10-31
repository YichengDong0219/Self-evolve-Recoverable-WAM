"""Simulate-and-select driver for hanging_mug recovery data (Claude, 2026-10-28). No model.

Every failure state is reproduced bit-exactly by replay, so candidates can be tried independently from the same
state. Per failure:
  1 list    MR6 with --mr-list under each grasp-rotation variant (RoboTwin's own first rotation "pp+seeds+rack";
            shortest-joint-path rotation "pp+seeds+best+rack") -> whole-chain-feasible (arm, contact point) pairs
  2 try     candidates in that order, each a fresh process from the replayed state (MR6 --mr-forced, single
            attempt, no homing, direct pick-and-hang), until one passes ALL quality gates (quality_gates.gates)
  3 keep    only that attempt is marked selected; every other attempt stays on disk as a rejected record
usage: python sim_select.py <jobs.jsonl> <out_dir> <gpu,gpu,...> [max_candidates=8] [repeats=1] [shard=k/n]
shard k/n (or k1,k2,.../n) processes only jobs whose line index % n is listed (several machines share one job list).
The GPU list may repeat a GPU to run several workers on it (a worker = one simulator, ~5 GB; mostly CPU-bound).
repeats > 1 tries each candidate again from the same replayed state when it failed (cuRobo's plans differ between
runs, so the physical outcome of one candidate varies; SELdev: a candidate selected twice failed in a third run).
Writes <out_dir>/<name>/{list_*.json, cand_*.json/.mp4/.log}, <out_dir>/<name>/selection.json,
<out_dir>/summary_<hostname>.jsonl (one file per machine so results can be merged by rsync).
"""
import json
import os
import queue
import socket
import subprocess
import sys
import threading

import quality_gates as QG

L = os.environ.get("FACT_LOCAL", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 10-30 (MR7 top-up): SIM_SELECT_JOINT_GATES=1 adds the pair-level joint gates to every attempt, so an attempt with a
# joint snap (G7, single-frame arm step > 0.2 rad) or an out-of-range frame (G8, asset G8 range) does not stop the
# search: the next candidate is tried. Same thresholds as mug_sft_lists.G7_MAX_STEP and the 10-30 audit.
JOINT_GATES = os.environ.get("SIM_SELECT_JOINT_GATES") == "1"
_ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def joint_gates(npz):
    import numpy as np
    meta = json.load(open(os.path.join(L, "assets/rf_hanging_mug_v1/meta.json")))
    lo, hi = np.array(meta["G8_range"]["lo"]), np.array(meta["G8_range"]["hi"])
    q = np.load(npz)["expert_state"]
    step = float(np.abs(np.diff(q[:, _ARM], axis=0)).max()) if len(q) > 1 else 0.0
    return {"G7_no_joint_snap": step <= 0.2, "G8_joint_range": not bool(((q < lo) | (q > hi))[:, _ARM].any())}
VARIANTS = ["pp+seeds+rack", "pp+seeds+best+rack"]
if os.environ.get("SIM_SELECT_VARIANTS"):  # MR7 (10-30): e.g. "pp+seeds+rack+jbox+ns+boxchk+flip,..."
    VARIANTS = os.environ["SIM_SELECT_VARIANTS"].split(",")


def bench(gpu, roll, step, variant, out, extra):
    cmd = (f"source {L}/env.sh {gpu}; timeout 2400 python {L}/code/takeover_bench.py --rollout {roll} --step {step} "
           f"--plan MR6 --variant {variant} --out {out} {extra} > {out[:-5]}.log 2>&1")
    subprocess.run(["bash", "-c", cmd])
    return json.load(open(out)) if os.path.exists(out) else None


def process(job, gpu, out_dir, max_cands, repeats=1):
    roll, step = job["rollout"], job["step"]
    seed = json.load(open(roll))["env_seed"]
    name = f'{job["group"]}_{seed}'
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    sel_path = os.path.join(d, "selection.json")
    if os.path.exists(sel_path):
        return json.load(open(sel_path))
    claim = os.path.join(d, ".claim")
    try:  # atomic: several sim_select instances on one machine share a job list without duplicate work
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{socket.gethostname()} {os.getpid()}".encode())
        os.close(fd)
    except FileExistsError:
        return None
    cands = []
    for vi, v in enumerate(VARIANTS):
        r = bench(gpu, roll, step, v, os.path.join(d, f"list_v{vi}.json"), "--mr-list")
        if r is None:  # crashed (e.g. a GPU dropped off CUDA, 10-29): no verdict; release the state for a later run
            os.remove(claim)
            return {"name": name, "error": f"list_v{vi} no output"}
        feas = next((e.get("feasible", []) for e in r.get("mr_log", []) if e["stage"] == "chain"), [])
        cands += [{"variant": v, "arm": f["arm"], "cid": f["cid"], "why": f["why"],
                   **({"flip": f["flip"]} if "flip" in f else {})} for f in feas]
    tried, selected = [], None
    sched = [(i, rep, c) for rep in range(repeats) for i, c in enumerate(cands[:max_cands])]
    for i, rep, c in sched:
        out = os.path.join(d, f"cand_{i:02d}.json" if rep == 0 else f"cand_{i:02d}_r{rep}.json")
        r = bench(gpu, roll, step, c["variant"], out,
                  f"--video --capture --mr-forced '{json.dumps({'arm': c['arm'], 'cid': c['cid'], **({'flip': c['flip']} if 'flip' in c else {})})}'")
        if r is None:  # crashed: same as above, never record a verdict built on a crash
            os.remove(claim)
            return {"name": name, "error": f"{os.path.basename(out)} no output"}
        g = QG.gates(r)
        npz = (r.get("capture") or {}).get("npz")
        if JOINT_GATES and npz and os.path.exists(npz):
            g.update(joint_gates(npz))
        ok = all(g.values()) and bool((r.get("capture") or {}).get("npz"))  # selected only with lossless SFT data
        tried.append({**c, "idx": i, "rep": rep, "status": r["status"], "layer": r["layer"], "gates": g, "pass": ok})
        if ok:
            selected = {**c, "idx": i, "rep": rep, "json": out, "video": out[:-5] + ".mp4"}
            break
    res = {"name": name, "rollout": roll, "step": step, "n_candidates": len(cands), "tried": tried,
           "selected": selected}
    json.dump(res, open(sel_path, "w"))
    return res


def main():
    jobs = [json.loads(line) for line in open(sys.argv[1])]
    if len(sys.argv) > 6:  # shard "k/n" or "k1,k2,.../n"
        ks, n = sys.argv[6].split("/")
        ks, n = {int(x) for x in ks.split(",")}, int(n)
        jobs = [j for i, j in enumerate(jobs) if i % n in ks]
    out_dir, gpus = sys.argv[2], sys.argv[3].split(",")
    max_cands = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    repeats = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    os.makedirs(out_dir, exist_ok=True)
    # release stale claims of dead sim_select processes on this machine (state not finished)
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

    def worker(gpu):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            r = process(j, gpu, out_dir, max_cands, repeats)
            if r is None:  # claimed by another instance
                continue
            if "error" in r:
                print(r["name"], "ERROR", r["error"], flush=True)
                continue
            with lock:
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
