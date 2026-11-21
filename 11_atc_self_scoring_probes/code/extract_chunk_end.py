"""ATC probe P2 input (Claude, 2026-11-21; user: "先做一些不用GPU/轻量GPU的探针实验"). CPU only.

For every chunk of the 11-14 chunk bank (wm_value/v2/chunk_rows.jsonl, 1682 rows), read the simulator log of the
rollout that executed it and keep the TRUE state at the end of the chunk (privileged: what a perfect imagination plus
perfect perception would see).
  policy chunk  <group>/runs/<state>__policy__<seed>.json, metrics[47] (or the last step if it finished earlier)
  expert chunk  <group>/runs/<state>__chunk1__20261130.json (the first 48 steps are the fixed cuRobo chunk), metrics[47]
Fields: dx, dy (mouse - pad, m), mouse height above the pad, orientation error, holder, gripper states, success flag.
usage: python extract_chunk_end.py <mouse_dpo_v3 dir> <chunk_rows.jsonl> <out.jsonl>
"""
import json
import sys
from pathlib import Path

ROOT, ROWS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
# The groups the 11-14 bank was built from = the ones named in wm_value/chunk_rates.jsonl (c1c is a later duplicate).
GROUP_DIRS = sorted({Path(json.loads(l)["group"]) for l in (ROOT / "wm_value" / "chunk_rates.jsonl").read_text().splitlines()
                     if l.strip()})


def main():
    where = {}
    for g in GROUP_DIRS:
        for b in json.loads((g / "bank.json").read_text()):
            if b["id"] in where:
                raise SystemExit(f"state {b['id']} in two groups")
            where[b["id"]] = g
    rows = [json.loads(l) for l in ROWS.read_text().splitlines() if l.strip()]
    out, missing = [], []
    for r in rows:
        g = where.get(r["state"])
        if g is None:
            missing.append(r["id"]); continue
        if r["kind"] == "policy":
            run = g / "runs" / f"{r['state']}__policy__{r['seed']}.json"
        else:
            run = g / "runs" / f"{r['state']}__chunk1__20261130.json"
        d = json.loads(run.read_text())
        if r["kind"] == "expert":
            assert all(p == "expert" for p in d["phases"][:48]), run
        m = d["metrics"]
        k = min(47, len(m) - 1)
        e = m[k]
        out.append(dict(id=r["id"], end_index=k, steps_in_log=len(m), dx=e["dx"], dy=e["dy"],
                        dz=e["mouse_p"][2] - e["pad_p"][2], orientation_error=e["orientation_error"],
                        holder=e["holder"], left_open=e["left_open"], right_open=e["right_open"],
                        success_end=bool(e["success"]), start_dx=d["initial_metrics"]["dx"],
                        start_dy=d["initial_metrics"]["dy"], start_holder=d["initial_metrics"]["holder"]))
    if missing:
        raise SystemExit(f"{len(missing)} chunks without a group, e.g. {missing[:3]}")
    OUT.write_text("".join(json.dumps(x) + "\n" for x in out))
    print(json.dumps(dict(chunks=len(out), success_end=sum(x["success_end"] for x in out),
                          held_end=sum(x["holder"] != "none" for x in out))))


if __name__ == "__main__":
    main()
