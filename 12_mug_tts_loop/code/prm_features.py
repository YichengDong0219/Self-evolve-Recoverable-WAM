"""PRM features for one TTS round (Claude, 2026-11-21; v2 2026-11-22: incremental, phi_re24 dropped). Client of
prm_service.py (any Python with numpy + PIL). For every episode <ep>.json with <ep>.tts.npz in the round dir it queries
whatever is still missing:
  trajectory states d:     phi(cur_d) [forward], hop(cur_{d-1} -> cur_d) [incremental, d >= 1]
  candidate i at state d:  imagined: hop(cur_d -> im48_i), phi(im48_i), phi(im24_i)
                           real, once <ep>.exec.{json,npz} exist: hop(cur_d -> re48_i), phi(re48_i)
Results accumulate in <ep>.prm.json; "complete" = exec output present and every candidate with an executed +48
keyframe has its real features. v1 also queried phi(re24), which no analysis uses; v2 no longer does.
Order: episodes whose exec output exists (they become complete) first, then imagined-only work; the list is rebuilt
after every episode so newly finished exec outputs jump the queue. Exits when nothing currently computable is
missing and prints "PRM_PASS_DONE incomplete=<n> without_tts_features=<m>".
All images follow prm_common's convention (FACT canvas sizes).
usage: python prm_features.py --round-dir <dir> [--addr 127.0.0.1:8650] [--batch 128]
"""
import argparse
import json
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prm_common import canvas_views, imagined_canvas, real_views  # noqa: E402

TRAJ = ("phi_cur", "hop_prev")
IMAG = ("hop_cur_to_im48", "phi_im48", "phi_im24")
REAL = ("hop_cur_to_re48", "phi_re48")


def load_prm(epf):
    p = epf.with_suffix(".prm.json")
    return json.loads(p.read_text()) if p.exists() else None


def episodes(rd):
    return [e for e in sorted(rd.glob("hanging_mug_seed*.json")) if e.name.count(".") == 1
            and e.with_suffix(".tts.npz").exists()]


def status(epf):
    """(has_exec, complete, has_tts_features) from the stored file only (cheap)."""
    pr = load_prm(epf)
    has_exec = epf.with_suffix(".exec.json").exists() and epf.with_suffix(".exec.npz").exists()
    return has_exec, bool(pr and pr.get("complete")), bool(pr and pr.get("tts_done"))


def plan(epf, pr):
    """Items still missing for this episode -> (items, keys, D, N, has_exec, n_real_expected)."""
    ep = json.loads(epf.read_text())
    z = np.load(epf.with_suffix(".tts.npz"))
    # every z[key] access decompresses the whole array again: materialize each array exactly once
    imgs, c_head, c_left, c_right = z["imgs"], z["cur"], z["cur_left"], z["cur_right"]
    exz = epf.with_suffix(".exec.npz")
    has_exec = epf.with_suffix(".exec.json").exists() and exz.exists()
    kz = np.load(exz) if has_exec else None
    kz_files = set(kz.files) if has_exec else set()
    feats = (pr or {}).get("features", {})
    have = lambda name, key: key in feats.get(name, {})  # noqa: E731
    task = ep["instruction"]
    D, N = imgs.shape[:2]
    cur = [real_views(c_head[d], c_left[d], c_right[d]) for d in range(D)]
    start = cur[0]
    items, keys, n_real = [], [], 0
    for d in range(D):
        if not have("phi_cur", str(d)):
            items.append(dict(task=task, start=start, before=start, after=cur[d])); keys.append(("phi_cur", str(d)))
        if d >= 1 and not have("hop_prev", str(d)):
            items.append(dict(task=task, start=start, before=cur[d - 1], after=cur[d])); keys.append(("hop_prev", str(d)))
        for i in range(N):
            key = f"{d}_{i}"
            if not all(have(n, key) for n in IMAG):
                im48 = canvas_views(imagined_canvas(imgs[d], i, 4))
                im24 = canvas_views(imagined_canvas(imgs[d], i, 2))
                items += [dict(task=task, start=start, before=cur[d], after=im48),
                          dict(task=task, start=start, before=start, after=im48),
                          dict(task=task, start=start, before=start, after=im24)]
                keys += [("hop_cur_to_im48", key), ("phi_im48", key), ("phi_im24", key)]
            kk = f"d{d}_c{i}"
            if has_exec and kk in kz_files:
                n_real += 1
                if not all(have(n, key) for n in REAL):
                    re48 = real_views(*kz[kk][3])               # [4 keyframes, 3 views, H, W, 3] -> +48
                    items += [dict(task=task, start=start, before=cur[d], after=re48),
                              dict(task=task, start=start, before=start, after=re48)]
                    keys += [("hop_cur_to_re48", key), ("phi_re48", key)]
    return ep, items, keys, D, N, has_exec, n_real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True); ap.add_argument("--addr", default="127.0.0.1:8650")
    ap.add_argument("--batch", type=int, default=128)
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

    tried = set()
    while True:
        todo = []
        for e in episodes(a.round_dir):
            if e.name in tried:
                continue
            has_exec, complete, tts_done = status(e)
            if has_exec and not complete:
                todo.append((0, e))
            elif not tts_done:
                todo.append((1, e))
        if not todo:
            break
        todo.sort(key=lambda x: (x[0], x[1].name))
        epf = todo[0][1]
        tried.add(epf.name)          # one visit per episode per pass; the caller re-runs the pass later
        pr = load_prm(epf)
        tp = time.time()
        ep, items, keys, D, N, has_exec, n_real = plan(epf, pr)
        t0 = time.time()
        sc = score(items) if items else []
        feats = (pr or {}).get("features", {})
        for (name, key), s in zip(keys, sc):
            feats.setdefault(name, {})[key] = s
        n_have_real = len(feats.get("phi_re48", {}))
        res = dict(episode=epf.name, decisions=int(D), n=int(N), version=2, features=feats,
                   queries=int((pr or {}).get("queries", 0)) + len(items), tts_done=True,
                   exec_seen=has_exec, complete=bool(has_exec and n_have_real >= n_real))
        tmp = epf.with_suffix(".prm.tmp")
        tmp.write_text(json.dumps(res))
        tmp.rename(epf.with_suffix(".prm.json"))
        dt = time.time() - t0
        print(f"{epf.name} +{len(items)} queries plan {t0 - tp:.0f}s score {dt:.0f}s "
              f"({len(items) / max(dt, 1e-6):.1f} q/s) complete={res['complete']}", flush=True)
    st = [status(e) for e in episodes(a.round_dir)]
    print(f"PRM_PASS_DONE incomplete={sum(1 for s in st if not s[1])} "
          f"without_tts_features={sum(1 for s in st if not s[2])}", flush=True)


if __name__ == "__main__":
    main()
