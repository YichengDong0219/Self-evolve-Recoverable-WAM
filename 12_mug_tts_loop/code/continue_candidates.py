"""Gate continuations (Claude, 2026-11-21; protocol 12, step 5). Runs in the RoboTwin env against a STANDARD probe
server (π0, count 1, fast mode: --enable_prefix_cache --skip_future_state_value). For each requested (decision d,
candidate i) of a TTS episode: fresh scene (episode seed + instruction), replay executed[:48d], verify the capture,
execute candidate i (48 steps or until success), then let π0 continue to the end (step_lim) with explicit
continuation seeds aligned to decision index d+1. Records the true outcome of that candidate under π0's continuation.
Output: <ep>.cont.jsonl (one row per pair; appends; a pair already present is skipped).
usage: python continue_candidates.py --episode <ep>.json --pairs d:i,d:i,... --port <probe server port>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import g0_driver as G  # noqa: E402
from exec_candidates import episode_info, same  # noqa: E402

PLAN = 48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=Path, required=True); ap.add_argument("--pairs", required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    out = a.episode.with_suffix(".cont.jsonl")
    done = set()
    if out.exists():
        done = {(r["decision"], r["cand"]) for r in map(json.loads, out.read_text().splitlines()) if r}
    ep = json.loads(a.episode.read_text())
    z = np.load(a.episode.with_suffix(".tts.npz"))
    cands = z["cands"]
    args = G.build_args(ep["task"], ep["config"])
    env = G.task_env(ep["task"])
    seed, info = ep["env_seed"], episode_info(env, args, ep)
    acts = ep["executed_actions"]
    for pr in a.pairs.split(","):
        d, i = map(int, pr.split(":"))
        if (d, i) in done:
            continue
        t0 = time.time()
        G.setup_episode(env, args, seed, info, forced=ep["instruction"])
        mi, policy = G.make_policy(a.port)
        mi.reset_model(policy)
        policy.bind_probe(ep["start_id"], ep["root_seed"], G.server_sink([]))
        for s in range(d * PLAN):
            env.get_obs(); env.take_action(np.asarray(acts[s], dtype=np.float32))
        ok, diff = same(G.capture(env), ep["states"][str(d * PLAN)])
        row = dict(decision=d, cand=i, start_ok=ok, start_maxdiff=diff)
        if ok:
            for t in range(PLAN):
                env.get_obs(); env.take_action(np.asarray(cands[d, i, t], dtype=np.float32))
                if env.eval_success:
                    break
            policy._probe_decision = d + 1          # continuation seeds aligned with the decision index
            while not env.eval_success and env.take_action_cnt < env.step_lim:
                obs = env.get_obs()
                mi.eval(env, policy, obs)
            row.update(success=bool(env.eval_success), steps=int(env.take_action_cnt))
        row["wall_s"] = round(time.time() - t0, 1)
        env.close_env()
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
