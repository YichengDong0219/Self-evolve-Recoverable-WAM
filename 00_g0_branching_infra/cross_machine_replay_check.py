"""Compare per-boundary simulator states of a local replay against the remote source rollout (Claude, 2026-10-28).

usage: python cross_machine_replay_check.py <source_rollout.json> <local_replay.json>
Prints, per stored boundary (0, 48, ..., final), whether every recorded field matches exactly and, if not, the
largest numeric deviation. Exit 0 only if all boundaries match bit-exactly.
"""
import json
import sys


def flatten(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flatten(v, f"{prefix}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def main():
    src, rep = (json.load(open(p)) for p in sys.argv[1:3])
    ok_all = src["executed_actions_sha256"] == rep["executed_actions_sha256"] and src["success"] == rep["success"]
    print("actions sha equal:", src["executed_actions_sha256"] == rep["executed_actions_sha256"],
          "| success", src["success"], rep["success"])
    for k in src["states"]:
        a, b = dict(flatten(src["states"][k])), dict(flatten(rep["states"].get(k, {})))
        if a == b:
            print(f"boundary {k}: exact")
            continue
        ok_all = False
        worst, n = 0.0, 0
        for key in set(a) | set(b):
            if a.get(key) != b.get(key):
                n += 1
                if isinstance(a.get(key), (int, float)) and isinstance(b.get(key), (int, float)):
                    worst = max(worst, abs(a[key] - b[key]))
        print(f"boundary {k}: {n} fields differ, max numeric deviation {worst:.3e}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
