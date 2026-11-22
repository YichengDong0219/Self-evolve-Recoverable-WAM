"""Smoke test of a running prm_service.py (Claude, 2026-11-22; protocol 12 step 2). Client; numpy + PIL.
Uses one finished round-0 episode that has offline-exec output and succeeded. Checks (all must hold, else rc 1):
  1. every reply parses as <score>..%</score> (no NaN) and lies in [-1, 1];
  2. wiring sanity: forward phi(start -> start) is about 0 (|phi| <= 0.2), and phi(start -> the success frame of
     the episode's last executed chunk) > phi(start -> start) + 0.3 (images in the right slots, task text used).
Also reports (informational, v2 11-22) how many scores change when the same batch is sent twice -- the first version
required bit-identical repeats, which batched vLLM does not give (prm_determinism.py) -- and the cold throughput.
Writes <round dir>/prm_smoke.json.
usage: python prm_smoke.py --round-dir <dir> [--episode <ep>.json] [--addr 127.0.0.1:8650]
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


def build_items(round_dir, epf=None):
    """The smoke-test query set of one successful episode with exec output -> (epf, ep, items, names, succ_row)."""
    if epf is None:
        for f in sorted(round_dir.glob("hanging_mug_seed*.exec.json")):
            cand = f.with_name(f.name.replace(".exec.json", ".json"))
            if cand.exists() and json.loads(cand.read_text())["success"]:
                epf = cand; break
    if epf is None:
        raise SystemExit("no successful episode with exec output yet")
    ep = json.loads(epf.read_text())
    z = np.load(epf.with_suffix(".tts.npz")); kz = np.load(epf.with_suffix(".exec.npz"))
    ex = json.loads(epf.with_suffix(".exec.json").read_text())
    task, D, N = ep["instruction"], z["cands"].shape[0], z["cands"].shape[1]
    cur = [real_views(z["cur"][d], z["cur_left"][d], z["cur_right"][d]) for d in range(D)]
    start = cur[0]
    last, ks = D - 1, int(z["k"][D - 1])
    succ_row = [r for r in ex["rows"] if r["decision"] == last and r["cand"] == ks][0]
    succ_frame = real_views(*kz[f"d{last}_c{ks}"][3])
    items, names = [dict(task=task, start=start, before=start, after=start)], ["phi_start"]
    items.append(dict(task=task, start=start, before=start, after=succ_frame)); names.append("phi_success_frame")
    for d in sorted({0, D // 2, last}):
        items.append(dict(task=task, start=start, before=start, after=cur[d])); names.append(f"phi_cur_d{d}")
        for i in range(N):
            im48 = canvas_views(imagined_canvas(z["imgs"][d], i, 4))
            items += [dict(task=task, start=start, before=cur[d], after=im48),
                      dict(task=task, start=start, before=start, after=im48)]
            names += [f"hop_im48_d{d}_c{i}", f"phi_im48_d{d}_c{i}"]
            kk = f"d{d}_c{i}"
            if kk in kz.files:
                re48 = real_views(*kz[kk][3])
                items += [dict(task=task, start=start, before=cur[d], after=re48),
                          dict(task=task, start=start, before=start, after=re48)]
                names += [f"hop_re48_d{d}_c{i}", f"phi_re48_d{d}_c{i}"]
    return epf, ep, items, names, succ_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True); ap.add_argument("--episode", type=Path)
    ap.add_argument("--addr", default="127.0.0.1:8650")
    a = ap.parse_args()
    epf, ep, items, names, succ_row = build_items(a.round_dir, a.episode)
    host, port = a.addr.split(":")
    conn = Client((host, int(port)), authkey=b"mug-tts-critic")
    runs = []
    for _ in range(2):
        t0 = time.time()
        conn.send({"op": "prm", "items": items})
        rep = conn.recv()
        if "error" in rep:
            raise SystemExit("service error: " + rep["error"])
        runs.append((rep["scores"], rep["raw"], time.time() - t0))
    s1, raw1, t1 = runs[0]; s2, _, t2 = runs[1]
    s1a, dd = np.asarray(s1, float), np.abs(np.asarray(s1, float) - np.asarray(s2, float))
    res = dict(episode=epf.name, success=ep["success"], succ_row_success=succ_row.get("success_within_chunk"),
               queries=len(items), seconds=[round(t1, 1), round(t2, 1)], qps_first_cold=round(len(items) / t1, 2),
               nan=int(np.isnan(s1a).sum()), in_range=bool(np.all(np.abs(s1a[~np.isnan(s1a)]) <= 1.0)),
               repeat_identical=bool(s1 == s2), repeat_items_differ=int((dd > 0).sum()),
               repeat_max_abs_diff=float(np.nanmax(dd)), raw_examples=raw1[:4],
               phi_start=s1[0], phi_success_frame=s1[1],
               scores={n: s for n, s in zip(names, s1)})
    # Bit-identical repeats are informational only: batched vLLM greedy decoding is not bit-reproducible here
    # (11-22 diagnosis: ~5% of items change by mean |d| ~0.001 with or without prefix caching; README).
    res["checks"] = dict(parse=res["nan"] == 0, range=res["in_range"],
                         phi_start_near_0=abs(s1[0]) <= 0.2, success_above_start=s1[1] > s1[0] + 0.3)
    res["pass"] = all(res["checks"].values())
    (a.round_dir / "prm_smoke.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "scores"}))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
