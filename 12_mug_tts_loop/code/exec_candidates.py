"""Offline execution of TTS candidates (Claude, 2026-11-21; user: "开始执行"). Protocol: 12_mug_tts_loop/PROTOCOL.md.
Runs in the RoboTwin env (cwd = RoboTwin-FACT). No model. For one TTS episode (tts_driver.py output: <ep>.json +
<ep>.tts.npz), for every decision d and every candidate i (all N, the executed one included):
  1. fresh scene with the episode's env seed (expert check once per process, as G0), same instruction;
  2. replay the episode's executed actions up to step 48d (take_action only; --render-prefix also renders every step
     like G0 replay -- the capture check in 3 decides whether skipping the renders changes anything);
  3. verify the physics/task capture at 48d equals the online capture (actors, articulations, RNG digests);
     a mismatch marks the pair invalid (counted, never used);
  4. execute the candidate's 48 actions (stop early on task success), rendering the 3 views after 12/24/36/48 actions;
  5. store keyframes (uint8, native resolution), the end capture (+48 or at success) and success_within_chunk.
Consistency check: for the executed candidate, the +48 capture must equal the online capture at 48(d+1).
Output: <ep>.exec.npz + <ep>.exec.json (refuses to overwrite).
usage: python exec_candidates.py --episode <ep>.json [--decisions 0,1,..] [--candidates all|nonselected]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import g0_driver as G  # noqa: E402

PLAN = 48
VIEWS = ("head_camera", "left_camera", "right_camera")


def frames(env):
    obs = env.get_obs()
    return [np.asarray(obs["observation"][v]["rgb"][..., :3], dtype=np.uint8) for v in VIEWS]


def flat(cap):
    """Numeric vector of a capture (actor poses/velocities, articulation qpos/qvel/drive targets) + RNG digests."""
    vals = []
    for k in sorted(cap["actors"]):
        r = cap["actors"][k]
        vals += r["p"] + r["q"] + r.get("v", []) + r.get("w", [])
    for k in sorted(cap["articulations"]):
        r = cap["articulations"][k]
        vals += r["root_p"] + r["root_q"] + r["qpos"] + r["qvel"]
        for t in r["drive_target"] + r["drive_velocity_target"]:
            vals += t
    return np.asarray(vals, dtype=np.float64), (cap["np_rng_sha"], cap["py_rng_sha"], cap["action_step"])


def same(a, b):
    va, ka = flat(a)
    vb, kb = flat(b)
    if va.shape != vb.shape or ka != kb:
        return False, float("inf")
    d = float(np.max(np.abs(va - vb))) if va.size else 0.0
    return d == 0.0, d


def episode_info(env, args, ep):
    """The expert-episode info G0's setup_episode needs (instruction wording). Stored by tts_driver as expert_info;
    otherwise (older episodes) play the expert once on the episode seed and keep its info whatever the plan outcome:
    expert validity is stochastic (planner), the scene info is not."""
    if ep.get("expert_info") is not None:
        return ep["expert_info"]
    for _ in range(3):
        try:
            env.setup_demo(now_ep_num=0, seed=ep["env_seed"], is_test=True, **args)
            info = env.play_once()
            env.close_env()
            return info
        except Exception:  # noqa: BLE001
            try:
                env.close_env()
            except Exception:  # noqa: BLE001
                pass
    raise RuntimeError("could not obtain the expert info")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--decisions", default="all"); ap.add_argument("--candidates", default="all")
    ap.add_argument("--render-prefix", action="store_true", help="call get_obs() on every prefix step like G0 replay")
    ap.add_argument("--out-suffix", default="exec")
    a = ap.parse_args()
    out_npz, out_json = a.episode.with_suffix(f".{a.out_suffix}.npz"), a.episode.with_suffix(f".{a.out_suffix}.json")
    if out_npz.exists() or out_json.exists():
        raise SystemExit("refusing to overwrite exec outputs")
    ep = json.loads(a.episode.read_text())
    z = np.load(a.episode.with_suffix(".tts.npz"))
    cands, ks = z["cands"], z["k"]
    D, N = cands.shape[0], cands.shape[1]
    decs = range(D) if a.decisions == "all" else [int(x) for x in a.decisions.split(",")]
    args = G.build_args(ep["task"], ep["config"])
    env = G.task_env(ep["task"])
    seed, info = ep["env_seed"], episode_info(env, args, ep)
    acts = ep["executed_actions"]
    rows, kf, t_all = [], {}, time.time()
    for d in decs:
        for i in range(N):
            if a.candidates == "nonselected" and i == int(ks[d]):
                continue
            t0 = time.time()
            G.setup_episode(env, args, seed, info, forced=ep["instruction"])
            for s in range(d * PLAN):
                if a.render_prefix:
                    env.get_obs()
                env.take_action(np.asarray(acts[s], dtype=np.float32))
            ok, diff = same(G.capture(env), ep["states"][str(d * PLAN)])
            row = dict(decision=d, cand=i, selected=bool(i == int(ks[d])), start_ok=ok, start_maxdiff=diff)
            if ok:
                ims, fin_at = [], None
                for t in range(PLAN):
                    env.get_obs()
                    env.take_action(np.asarray(cands[d, i, t], dtype=np.float32))
                    if env.eval_success and fin_at is None:
                        fin_at = t + 1
                    if (t + 1) % 12 == 0 or (env.eval_success and len(ims) < 4):
                        while len(ims) < (t + 1) // 12 or (env.eval_success and len(ims) < 4):
                            ims.append(frames(env))
                    if env.eval_success:
                        break
                cap = G.capture(env)
                kf[f"d{d}_c{i}"] = np.stack([np.stack(v) for v in ims[:4]]) if len(ims[0][0].shape) == 3 else None
                row.update(success_within_chunk=bool(env.eval_success), finished_at=fin_at, end_capture=cap)
                if i == int(ks[d]) and str((d + 1) * PLAN) in ep["states"] and fin_at is None:
                    c_ok, c_diff = same(cap, ep["states"][str((d + 1) * PLAN)])
                    row.update(consistency_ok=c_ok, consistency_maxdiff=c_diff)
            row["wall_s"] = round(time.time() - t0, 1)
            rows.append(row)
            env.close_env()
            print(json.dumps({k: v for k, v in row.items() if k != "end_capture"}), flush=True)
    np.savez_compressed(out_npz, **{k: v for k, v in kf.items() if v is not None})
    out_json.write_text(json.dumps(dict(episode=str(a.episode), env_seed=seed, decisions=D, n=N, rows=rows,
                                        wall_s=round(time.time() - t_all, 1))))
    print("EXEC_DONE", json.dumps(dict(pairs=len(rows), start_ok=sum(r["start_ok"] for r in rows),
                                       consistency_ok=sum(r.get("consistency_ok", False) for r in rows))))


if __name__ == "__main__":
    main()
