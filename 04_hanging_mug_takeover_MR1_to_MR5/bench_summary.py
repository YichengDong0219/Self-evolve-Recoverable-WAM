"""Summarise a takeover-benchmark variant (Claude, 2026-10-28).

usage: python bench_summary.py <bench/variant dir> [<bench/other variant dir> ...]
Per group: episodes, episodes recovered by any plan, per-plan success, and the failure-layer distribution of
attempts that were actually executed (not blocked / not applicable). With two dirs, also per-episode flips.
"""
import collections
import glob
import json
import os
import sys


def load(d):
    rows = []
    for p in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(p))
        r["group"] = os.path.basename(p).split("_1")[0]
        r["episode"] = (r["group"], r["env_seed"], r["step"])
        rows.append(r)
    return rows


def summary(rows, name):
    print(f"=== {name}: {len(rows)} attempts")
    by_g = collections.defaultdict(list)
    for r in rows:
        by_g[r["group"]].append(r)
    for g, rs in sorted(by_g.items()):
        eps = {r["episode"] for r in rs}
        rec = {r["episode"] for r in rs if r["status"] == "SUCCESS"}
        plans = collections.defaultdict(lambda: [0, 0, 0])
        for r in rs:
            c = plans[r["plan"]]
            c[0] += r["status"] == "SUCCESS"
            c[1] += r["status"] in ("SUCCESS", "FAIL")
            c[2] += 1
        layers = collections.Counter(r["layer"] if r["status"] != "SUCCESS" else "success" for r in rs
                                     if r["status"] in ("SUCCESS", "FAIL"))
        blocked = collections.Counter(r["layer"] for r in rs if r["status"] == "not_applicable")
        print(f"[{g}] episodes {len(eps)}  recovered by any plan {len(rec)}")
        print("   per plan success/executed/total: " + ", ".join(f"{p} {a}/{b}/{c}" for p, (a, b, c) in sorted(plans.items())))
        print("   layers: " + ", ".join(f"{k} {v}" for k, v in layers.most_common()))
        if blocked:
            print("   not executed: " + ", ".join(f"{k} {v}" for k, v in blocked.most_common()))
    return {r["episode"] for r in rows if r["status"] == "SUCCESS"}, {r["episode"] for r in rows}


def main():
    res = [summary(load(d), os.path.basename(d.rstrip("/"))) for d in sys.argv[1:]]
    if len(res) == 2:
        (a, ea), (b, eb) = res
        common = ea & eb
        print(f"=== flips on {len(common)} common episodes: gained {len((b - a) & common)}, lost {len((a - b) & common)}")


if __name__ == "__main__":
    main()
