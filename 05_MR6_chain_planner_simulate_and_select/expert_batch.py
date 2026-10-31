"""Expert demos for every selected hanging_mug recovery state (Claude, 2026-10-29). No model.

usage: python expert_batch.py <SEL dir> <out dir> <gpu,gpu,...>
For each <SEL dir>/*/selection.json with a selected candidate, runs collect_expert_demo.py on the same FACT rollout
(same scene seed + instruction) -> <out dir>/<state name>.{json,npz,log}. Existing outputs are skipped, so the batch
can be rerun as more states get selected. The GPU list may repeat a GPU for several workers.
"""
import glob
import json
import os
import queue
import subprocess
import sys
import threading

L = os.environ.get("FACT_LOCAL", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    sel, out, gpus = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    os.makedirs(out, exist_ok=True)
    q = queue.Queue()
    for p in sorted(glob.glob(os.path.join(sel, "*", "selection.json"))):
        r = json.load(open(p))
        if r.get("selected") and not os.path.exists(os.path.join(out, r["name"] + ".json")):
            q.put(r)

    def worker(gpu):
        while True:
            try:
                r = q.get_nowait()
            except queue.Empty:
                return
            stem = os.path.join(out, r["name"])
            subprocess.run(["bash", "-c", f"source {L}/env.sh {gpu}; timeout 1800 python {L}/code/collect_expert_demo.py "
                                          f"--rollout {r['rollout']} --out {stem} > {stem}.log 2>&1"])
            st = json.load(open(stem + ".json"))["status"] if os.path.exists(stem + ".json") else "NO_OUTPUT"
            print(r["name"], st, flush=True)

    ts = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("done")


if __name__ == "__main__":
    main()
