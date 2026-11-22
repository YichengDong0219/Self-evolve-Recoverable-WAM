"""Frozen-critic scores of the N = 32 snapshot candidates (Claude, 2026-11-22; PROTOCOL addendum 4).
Client of prm_service.py started with --critic = the Q part of r0/critic_gate.json (the gate critic, trained on the 80
non-held-out scenes; the 20 snapshot scenes are its held-out scenes). Per state one "score" request with all 32
candidates, exactly what the round-1 TTS driver would send (tts_driver.make_picker): instruction, start frames = the
episode's decision-0 request frames, current request frames, state, candidate actions, imagined keyframes. The service
computes PRM hop(cur -> im48), phi(im48), phi(im24) per candidate plus the action statistics and the linear Q score.
Output: <gen-dir>/n32_scores.jsonl, one row per state (scores [32], feats [32][7], service seconds); skips states done.
Any python with numpy (multiprocessing.connection client). usage: python n32_score.py --gen-dir <dir> [--addr 127.0.0.1:8650]
"""
import argparse
import json
import re
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", type=Path, required=True); ap.add_argument("--addr", default="127.0.0.1:8650")
    a = ap.parse_args()
    out = a.gen_dir / "n32_scores.jsonl"
    done = {(r["seed"], r["decision"]) for r in map(json.loads, out.read_text().splitlines()) if r} if out.exists() else set()
    instr = {}
    for f in a.gen_dir.glob("gen_seed*.jsonl"):
        for r in map(json.loads, f.read_text().splitlines()):
            if r.get("start_ok"):
                instr[(int(f.name[8:14]), r["decision"])] = r["instruction"]
    host, port = a.addr.split(":")
    conn = Client((host, int(port)), authkey=b"mug-tts-critic")
    files = sorted(a.gen_dir.glob("seed*_d*.n32.npz"))
    for f in files:
        s, d = map(int, re.match(r"seed(\d+)_d(\d+)\.n32\.npz", f.name).groups())
        if (s, d) in done:
            continue
        z = np.load(f)
        z0 = np.load(a.gen_dir / f"seed{s}_d0.n32.npz")
        msg = dict(op="score", instruction=instr[(s, d)], start=[z0["cur_head"], z0["cur_left"], z0["cur_right"]],
                   cur=[z["cur_head"], z["cur_left"], z["cur_right"]], state=z["state"], cands=z["cands"], imgs=z["imgs"])
        conn.send(msg)
        rep = conn.recv()
        if "error" in rep:
            raise RuntimeError(f"critic service error at {s}:{d}: {rep['error']}")
        with out.open("a") as fo:
            fo.write(json.dumps(dict(seed=s, decision=d, scores=[float(x) for x in rep["scores"]],
                                     feats=[[float(v) for v in row] for row in rep["feats"]], s=rep["s"])) + "\n")
        print(s, d, rep["s"], flush=True)
    conn.close()
    print("N32_SCORE_DONE", len(files))


if __name__ == "__main__":
    main()
