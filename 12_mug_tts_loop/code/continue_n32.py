"""Continuations of the N = 32 snapshot candidates (Claude, 2026-11-22; PROTOCOL addendum 4). = continue_record.py
with K = 1 (the gate continuation: the candidate's chunk, then pi0 with the shared seeds, repeat 0), candidates read
from n32_generate.py's <cands-dir>/seed<S>_d<d>.cands.npz, a step cap (default 450; all 434 successes seen so far
ended by episode step 346) and no camera recording. Runs in the RoboTwin env against a standard probe server
(fast mode; one server may serve several drivers, inference_server_multi.py).
For (decision d, candidate i): fresh scene, replay executed[:48d], verify the capture, execute candidate i (48 steps,
get_obs every step, stop on success), then pi0 continues until success or the cap.
Output: <out-dir>/seed<S>.n32.jsonl (one row per pair; appends; a pair already present is skipped) and
<out-dir>/rec/seed<S>_d<d>_c<i>_n32.npz (privileged state every 12 steps, continuation actions).
usage: python continue_n32.py --episode <ep>.json --pairs d:i,... --port <port> --cands-dir <dir> --out-dir <dir>
"""
import argparse
import hashlib
import json
import socket
import time
from pathlib import Path

import numpy as np

import g0_driver as G  # noqa: E402
from continue_record import feats  # noqa: E402
from exec_candidates import episode_info, same  # noqa: E402

PLAN, EVERY = 48, 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=Path, required=True); ap.add_argument("--pairs", required=True)
    ap.add_argument("--port", type=int, required=True); ap.add_argument("--cands-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--max-steps", type=int, default=450)
    a = ap.parse_args()
    (a.out_dir / "rec").mkdir(parents=True, exist_ok=True)
    ep = json.loads(a.episode.read_text())
    seed = ep["env_seed"]
    out = a.out_dir / f"seed{seed}.n32.jsonl"
    done = set()
    if out.exists():
        done = {(r["decision"], r["cand"]) for r in map(json.loads, out.read_text().splitlines()) if r}
    args = G.build_args(ep["task"], ep["config"])
    env = G.task_env(ep["task"])
    info = episode_info(env, args, ep)
    acts = ep["executed_actions"]
    mi, policy, cache = None, None, {}
    for pr in a.pairs.split(","):
        d, i = map(int, pr.split(":"))
        if (d, i) in done:
            continue
        if d not in cache:
            cache[d] = np.load(a.cands_dir / f"seed{seed}_d{d}.cands.npz")["cands"]
        cand = cache[d][i]
        t0 = time.time()
        G.setup_episode(env, args, seed, info, forced=ep["instruction"])
        if policy is None:
            mi, policy = G.make_policy(a.port)
        mi.reset_model(policy)
        policy.bind_probe(ep["start_id"], ep["root_seed"], G.server_sink([]))
        for s in range(d * PLAN):
            env.take_action(np.asarray(acts[s], dtype=np.float32))
        ok, diff = same(G.capture(env), ep["states"][str(d * PLAN)])
        row = dict(decision=d, cand=i, start_ok=ok, start_maxdiff=diff, max_steps=a.max_steps)
        if ok:
            base = d * PLAN
            lim = min(int(env.step_lim), a.max_steps)
            caps = [feats(env, 0)]
            for t in range(PLAN):
                env.get_obs(); env.take_action(np.asarray(cand[t], dtype=np.float32))
                rel = env.take_action_cnt - base
                if rel % EVERY == 0:
                    caps.append(feats(env, rel))
                if env.eval_success:
                    break
            policy._probe_decision = d + 1          # continuation seeds aligned with the decision index (as the gate)
            acts_c = []
            orig = env.take_action

            def rec(action, action_type="qpos", _orig=orig):
                acts_c.append(np.asarray(action, dtype=np.float32).tobytes())
                res = _orig(action, action_type=action_type)
                rl = env.take_action_cnt - base
                if rl % EVERY == 0:
                    caps.append(feats(env, rl))
                return res
            env.take_action = rec
            try:
                while not env.eval_success and env.take_action_cnt < lim:
                    mi.eval(env, policy, env.get_obs())
            finally:
                del env.take_action
            rel = env.take_action_cnt - base
            if caps[-1][0] != rel:
                caps.append(feats(env, rel))
            row.update(success=bool(env.eval_success), steps=int(env.take_action_cnt),
                       success_rel=int(rel) if env.eval_success else None, capped=bool(not env.eval_success and env.take_action_cnt >= lim
                                                                                        and lim < int(env.step_lim)),
                       cont_actions=len(acts_c), cont_actions_sha=hashlib.sha256(b"".join(acts_c)).hexdigest()[:16])
            act = (np.frombuffer(b"".join(acts_c), dtype=np.float32).reshape(len(acts_c), -1) if acts_c
                   else np.zeros((0, 14), np.float32))
            np.savez_compressed(a.out_dir / "rec" / f"seed{seed}_d{d}_c{i}_n32.npz", caps=np.asarray(caps), actions=act)
        row["wall_s"] = round(time.time() - t0, 1)
        row["host"] = socket.gethostname()
        env.close_env()
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
