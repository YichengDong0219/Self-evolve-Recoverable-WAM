"""How reproducible are the PRM scores of a running prm_service.py? (Claude, 2026-11-22; diagnosis after the first
smoke test found the same batch scored twice gave different numbers). Client; numpy + PIL.
Scores the smoke-test query set (prm_smoke.build_items) several ways and compares item by item:
  A1, A2  the full set as one batch, twice in a row;
  B       the full set in a shuffled order (numpy default_rng(0)), mapped back;
  C       batches of 48 in the original order (as prm_features.py sends them);
  D       one item per request, first 24 items.
Writes <round dir>/prm_determinism_<tag>.json.
usage: python prm_determinism.py --round-dir <dir> --tag <name> [--addr 127.0.0.1:8650]
"""
import argparse
import json
import sys
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prm_smoke import build_items  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--addr", default="127.0.0.1:8650")
    a = ap.parse_args()
    epf, ep, items, names, _ = build_items(a.round_dir)
    host, port = a.addr.split(":")
    conn = Client((host, int(port)), authkey=b"mug-tts-critic")

    def score(batch):
        conn.send({"op": "prm", "items": batch})
        rep = conn.recv()
        if "error" in rep:
            raise SystemExit("service error: " + rep["error"])
        return rep["scores"]

    runs = {"A1": score(items), "A2": score(items)}
    perm = np.random.default_rng(0).permutation(len(items))
    sb = score([items[i] for i in perm]); b = [None] * len(items)
    for j, i in enumerate(perm):
        b[i] = sb[j]
    runs["B"] = b
    runs["C"] = sum((score(items[k:k + 48]) for k in range(0, len(items), 48)), [])
    runs["D"] = [score([it])[0] for it in items[:24]]

    def cmp(x, y):
        n = min(len(x), len(y))
        d = np.abs(np.asarray(x[:n], float) - np.asarray(y[:n], float))
        ex = [dict(name=names[i], a=x[i], b=y[i]) for i in np.flatnonzero(d > 0)[:6]]
        return dict(items=n, differ=int((d > 0).sum()), max_abs=float(d.max()), mean_abs=float(d.mean()), examples=ex)
    res = dict(tag=a.tag, episode=epf.name, queries=len(items),
               compare={"A1_vs_A2": cmp(runs["A1"], runs["A2"]), "A2_vs_B": cmp(runs["A2"], runs["B"]),
                        "A2_vs_C": cmp(runs["A2"], runs["C"]), "A2_vs_D": cmp(runs["A2"], runs["D"]),
                        "C_vs_D": cmp(runs["C"], runs["D"])},
               score_spread=dict(min=float(np.min(runs["A2"])), max=float(np.max(runs["A2"])),
                                 distinct=len(set(runs["A2"]))))
    (a.round_dir / f"prm_determinism_{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["compare"]))


if __name__ == "__main__":
    main()
