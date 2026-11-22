"""Offline simulation of "PRM progress drop -> roll back -> TTS + critic" (Claude, 2026-11-22 ~17:05; user proposal:
"当PRM检测到完成度下降时…退回到上一个snapshot，做tts+critic打分"). Post hoc, descriptive, the 20 dev scenes.
Data (all existing): the round-0 executed trajectory of each scene with the PRM on its real frame at every decision
(phi_cur, hop_prev); at every decision the 8 recorded candidates, each executed 48 steps offline with the PRM on the
real +48 frame (phi_re48, hop_cur_to_re48), and each continued by pi0 to the end (gate, one shared-seed continuation).
Pipeline per scene: trigger = first decision d whose PRM signal dropped (rules fixed before looking at the result:
T1 hop_prev <= -0.2; T2 phi_cur <= running max - 0.15); roll back k = 1, 2, 3 chunks (snapshot d - k); pick among its
8 candidates by (a) random, (b) PRM phi on the real +48 frame, (c) PRM hop current -> real +48, (d) oracle; outcome =
the picked candidate's gate continuation (ties: mean over the tied candidates).
Caveat: one continuation per candidate; outcomes that depend on later seeds make (b)-(d) optimistic.
usage (in mug_loop_1121): python code/prm_rollback_sim.py r0
"""
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

rd = Path(sys.argv[1])
cont = collections.defaultdict(dict)
for f in sorted(glob.glob(str(rd / "hanging_mug_seed*.cont.jsonl"))):
    s = int(Path(f).name[16:22])
    for r in map(json.loads, open(f).read().splitlines()):
        if r.get("start_ok") and "success" in r:
            cont[s][(r["decision"], r["cand"])] = float(r["success"])


def pick(T, S):
    top = np.isclose(S, np.max(S))
    return float(T[top].mean())


def triggers(p, D):
    out = {}
    phi = [p["phi_cur"][str(d)] for d in range(D)]
    hop = p.get("hop_prev", {})
    out["T1"] = next((d for d in range(1, D) if hop.get(str(d)) is not None and hop[str(d)] <= -0.2), None)
    out["T2"] = next((d for d in range(1, D) if phi[d] <= max(phi[:d]) - 0.15), None)
    return out


lines = ["PRM-drop rollback simulation (post hoc, descriptive; 20 dev scenes)"]
tot = collections.defaultdict(lambda: collections.defaultdict(float))
for s in sorted(cont):
    ep = json.load(open(rd / f"hanging_mug_seed{s}.json"))
    p = json.load(open(rd / f"hanging_mug_seed{s}.prm.json"))["features"]
    D = len([d for d in range(40) if str(d) in p["phi_cur"] and all((d, c) in cont[s] for c in range(8))])
    okd = [d for d in range(D) if any(cont[s][(d, c)] for c in range(8))]
    last_ok = max(okd) if okd else None
    tr = triggers(p, D)
    ok = bool(ep["success"])
    lines.append(f"{s} {'OK  ' if ok else 'FAIL'} last decision with a successful candidate {last_ok} | "
                 f"phi_cur {' '.join('%.2f' % p['phi_cur'][str(d)] for d in range(min(D, 10)))} | triggers {tr}")
    for tn, dt in tr.items():
        tot[tn]["scenes_ok" if ok else "scenes_fail"] += 1
        if dt is None:
            tot[tn]["no_trigger_ok" if ok else "no_trigger_fail"] += 1
            continue
        tot[tn]["trigger_ok" if ok else "trigger_fail"] += 1
        for k in (1, 2, 3):
            dr = dt - k
            if dr < 0:
                continue
            T = np.array([cont[s][(dr, c)] for c in range(8)])
            phi = np.array([p["phi_re48"][f"{dr}_{c}"] for c in range(8)])
            hop = np.array([p["hop_cur_to_re48"][f"{dr}_{c}"] for c in range(8)])
            res = dict(random=float(T.mean()), prm_phi=pick(T, phi), prm_hop=pick(T, hop), oracle=float(T.max()))
            for m, v in res.items():
                tot[tn][f"k{k}_{'ok' if ok else 'fail'}_{m}"] += v
            lines.append(f"   {tn} fires at d{dt}; roll back {k} -> d{dr}: successful candidates {int(T.sum())}/8 | "
                         + " ".join(f"{m} {v:.2f}" for m, v in res.items()))
lines.append("")
for tn, t in tot.items():
    lines.append(f"{tn}: fires in {int(t['trigger_fail'])}/{int(t['scenes_fail'])} failed scenes and "
                 f"{int(t['trigger_ok'])}/{int(t['scenes_ok'])} successful scenes")
    for k in (1, 2, 3):
        lines.append(f"   roll back {k}: expected rescued failed scenes " + ", ".join(
            f"{m} {t[f'k{k}_fail_{m}']:.2f}" for m in ("random", "prm_phi", "prm_hop", "oracle"))
            + " | successful scenes kept after a false trigger " + ", ".join(
            f"{m} {t[f'k{k}_ok_{m}']:.2f}" for m in ("random", "prm_phi", "prm_hop", "oracle")))
print("\n".join(lines))
