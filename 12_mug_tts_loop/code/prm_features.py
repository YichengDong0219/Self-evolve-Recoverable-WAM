"""PRM features for one TTS round (Claude, 2026-11-21; protocol 12). Client of prm_service.py (any Python with
numpy + PIL). For every episode <ep>.json with <ep>.tts.npz and <ep>.exec.{json,npz} in the round dir, queries:
  trajectory states d:     phi(cur_d) [forward], hop(cur_{d-1} -> cur_d) [incremental, d >= 1]
  candidate i at state d:  imagined: hop(cur_d -> im48_i), phi(im48_i), phi(im24_i)
                           real:     hop(cur_d -> re48_i), phi(re48_i), phi(re24_i)   (offline-executed keyframes)
All images follow prm_common's convention (FACT canvas sizes). Output <ep>.prm.json (refuses to overwrite).
usage: python prm_features.py --round-dir <dir> [--addr 127.0.0.1:8650] [--batch 48]
"""
import argparse
import json
import sys
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prm_common import canvas_views, imagined_canvas, real_views  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True); ap.add_argument("--addr", default="127.0.0.1:8650")
    ap.add_argument("--batch", type=int, default=48)
    a = ap.parse_args()
    host, port = a.addr.split(":")
    conn = Client((host, int(port)), authkey=b"mug-tts-critic")

    def score(items):
        out = []
        for b in range(0, len(items), a.batch):
            conn.send({"op": "prm", "items": items[b:b + a.batch]})
            rep = conn.recv()
            if "error" in rep:
                raise RuntimeError(rep["error"])
            out += rep["scores"]
        return out

    for epf in sorted(a.round_dir.glob("hanging_mug_seed*.json")):
        if epf.name.endswith((".exec.json", ".prm.json")):
            continue
        outp = epf.with_suffix(".prm.json")
        exj, exz = epf.with_suffix(".exec.json"), epf.with_suffix(".exec.npz")
        if outp.exists() or not exj.exists():
            continue
        ep = json.loads(epf.read_text())
        z = np.load(epf.with_suffix(".tts.npz"))
        ex = json.loads(exj.read_text())
        kz = np.load(exz)
        task = ep["instruction"]
        D, N = z["cands"].shape[:2]
        cur = [real_views(z["cur"][d], z["cur_left"][d], z["cur_right"][d]) for d in range(D)]
        start = cur[0]
        items, keys = [], []
        for d in range(D):
            items.append(dict(task=task, start=start, before=start, after=cur[d])); keys.append(("phi_cur", d, -1))
            if d >= 1:
                items.append(dict(task=task, start=start, before=cur[d - 1], after=cur[d])); keys.append(("hop_prev", d, -1))
            for i in range(N):
                im48, im24 = canvas_views(imagined_canvas(z["imgs"][d], i, 4)), canvas_views(imagined_canvas(z["imgs"][d], i, 2))
                items += [dict(task=task, start=start, before=cur[d], after=im48),
                          dict(task=task, start=start, before=start, after=im48),
                          dict(task=task, start=start, before=start, after=im24)]
                keys += [("hop_cur_to_im48", d, i), ("phi_im48", d, i), ("phi_im24", d, i)]
                kk = f"d{d}_c{i}"
                if kk in kz.files:
                    kf = kz[kk]                      # [4 keyframes, 3 views, H, W, 3] native resolution
                    re48, re24 = real_views(*kf[3]), real_views(*kf[1])
                    items += [dict(task=task, start=start, before=cur[d], after=re48),
                              dict(task=task, start=start, before=start, after=re48),
                              dict(task=task, start=start, before=start, after=re24)]
                    keys += [("hop_cur_to_re48", d, i), ("phi_re48", d, i), ("phi_re24", d, i)]
        sc = score(items)
        res = {}
        for (name, d, i), s in zip(keys, sc):
            res.setdefault(name, {})[f"{d}" if i < 0 else f"{d}_{i}"] = s
        tmp = outp.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(episode=epf.name, decisions=D, n=N, queries=len(items), features=res)))
        tmp.rename(outp)
        print(epf.name, "queries", len(items), flush=True)


if __name__ == "__main__":
    main()
