"""PRM features for the multi-chunk unit measurement (Claude, 2026-11-22; PROTOCOL addendum 3). Client of
prm_service.py (PRM venv). For every recorded pair (unit_measure/rec/*.npz) and every recorded plan boundary
rel = 48j <= --max-rel: phi(rel) [forward: BEFORE episode start, AFTER frame] and hop(cur_d -> rel) [incremental:
BEFORE the decision frame, AFTER frame]. Same image convention and task text as prm_features.py. Identical image
pairs (e.g. the +48 frame shared by K = 1/2/3, the +96 frame shared by K = 2/3) are queried once.
Output: unit_measure/unit_prm.json {"<seed>_<d>_<i>_K<K>_r<rel>": [phi, hop]}.
usage: python unit_prm.py --round-dir r0 --out-dir r0/unit_measure [--addr 127.0.0.1:8650] [--max-rel 192]
"""
import argparse
import hashlib
import json
import re
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prm_common import real_views  # noqa: E402


def h(views):
    return hashlib.sha256(b"".join(np.ascontiguousarray(v).tobytes() for v in views)).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--addr", default="127.0.0.1:8650"); ap.add_argument("--max-rel", type=int, default=192)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    host, port = a.addr.split(":")
    conn = Client((host, int(port)), authkey=b"mug-tts-critic")
    outf = a.out_dir / "unit_prm.json"
    res = json.loads(outf.read_text()) if outf.exists() else {}
    cache_ep = {}
    uniq, want = {}, []                     # query hash -> item; (result key, kind, query hash)
    for f in sorted((a.out_dir / "rec").glob("seed*_d*_c*_u*.npz")):
        s, d, i, K = map(int, re.match(r"seed(\d+)_d(\d+)_c(\d+)_u(\d+)\.npz", f.name).groups())
        if s not in cache_ep:
            ep = json.loads((a.round_dir / f"hanging_mug_seed{s}.json").read_text())
            zn = np.load(a.round_dir / f"hanging_mug_seed{s}.tts.npz")
            cur = [zn["cur"], zn["cur_left"], zn["cur_right"]]
            cache_ep[s] = (ep["instruction"], cur)
        task, cur = cache_ep[s]
        start = real_views(cur[0][0], cur[1][0], cur[2][0])
        cur_d = real_views(cur[0][d], cur[1][d], cur[2][d])
        z = np.load(f)
        rels = [int(r) for r in z["img_rel"]]
        heads, lefts, rights = (z["head"], z["left"], z["right"]) if rels else (None, None, None)
        for j, rel in enumerate(rels):
            if rel > a.max_rel:
                continue
            key = f"{s}_{d}_{i}_K{K}_r{rel}"
            if key in res:
                continue
            after = [heads[j], lefts[j], rights[j]]
            for kind, before in (("phi", start), ("hop", cur_d)):
                qh = h([*start, *before, *after]) + task
                uniq.setdefault(qh, dict(task=task, start=start, before=before, after=after))
                want.append((key, kind, qh))
    keys = list(uniq)
    print(f"{len(want)} feature values, {len(keys)} unique queries", flush=True)
    score = {}
    t0 = time.time()
    for b in range(0, len(keys), a.batch):
        chunk = keys[b:b + a.batch]
        conn.send({"op": "prm", "items": [uniq[k] for k in chunk]})
        r = conn.recv()
        if "error" in r:
            raise RuntimeError(r["error"])
        for k, v in zip(chunk, r["scores"]):
            score[k] = v
        print(f"  {b + len(chunk)}/{len(keys)} queries, {time.time() - t0:.0f}s", flush=True)
    for key, kind, qh in want:
        v = res.setdefault(key, [None, None])
        v[0 if kind == "phi" else 1] = score[qh]
    outf.write_text(json.dumps(res))
    print(f"UNIT_PRM_DONE {len(res)} keys", flush=True)


if __name__ == "__main__":
    main()
