"""Gate continuations (Claude, 2026-11-21; protocol 12, step 5). Runs in the RoboTwin env against a STANDARD probe
server (π0, count 1, fast mode: --enable_prefix_cache --skip_future_state_value). For each requested (decision d,
candidate i) of a TTS episode: fresh scene (episode seed + instruction), replay executed[:48d], verify the capture,
execute candidate i (48 steps or until success), then let π0 continue to the end (step_lim) with explicit
continuation seeds aligned to decision index d+1. Records the true outcome of that candidate under π0's continuation.
Output: <ep>.cont.jsonl (one row per pair; appends; a pair already present is skipped).
--fast-obs (11-22): FACT's own fast-eval client path (_patch_robotwin_low_frequency_rgb: full RGB observations only at
plan boundaries, light observations otherwise, which FACT's eval() never reads; take_action skips the renderer pose
sync). Each row also records a hash of the continuation's executed actions, so two variants can be compared.
usage: python continue_candidates.py --episode <ep>.json --pairs d:i,d:i,... --port <probe server port> [--fast-obs]
"""
import argparse
import hashlib
import json
import socket
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
    ap.add_argument("--render-prefix", action="store_true",
                    help="call get_obs() on every prefix step (off by default: exec_candidates verified that skipping it "
                         "leaves the captured state and the later frames bit-identical)")
    ap.add_argument("--fast-obs", action="store_true", help="FACT fast-eval client path (see module docstring)")
    ap.add_argument("--out-suffix", default="cont")
    a = ap.parse_args()
    out = a.episode.with_suffix(f".{a.out_suffix}.jsonl")
    if a.fast_obs:
        from envs._base_task import Base_Task
        from evaluation.robotwin.model2robotwin_interface import _patch_robotwin_low_frequency_rgb
        _patch_robotwin_low_frequency_rgb()
        Base_Task._fact_low_frequency_rgb_enabled = True
        Base_Task._fact_execute_actions_per_plan = PLAN
        Base_Task._fact_skip_action_render_sync = True
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
    # ONE client per process: the FACT server (world_action_model.sockets, listen backlog 1) serves one connection
    # at a time until it closes, so a second client in the same process would wait forever (11-22 00:49 test hang).
    # ProbeClient supports reset + bind_probe per pair.
    mi, policy = None, None
    for pr in a.pairs.split(","):
        d, i = map(int, pr.split(":"))
        if (d, i) in done:
            continue
        t0 = time.time()
        G.setup_episode(env, args, seed, info, forced=ep["instruction"])
        if policy is None:
            mi, policy = G.make_policy(a.port)
        mi.reset_model(policy)
        policy.bind_probe(ep["start_id"], ep["root_seed"], G.server_sink([]))
        for s in range(d * PLAN):
            if a.render_prefix:
                env.get_obs()
            env.take_action(np.asarray(acts[s], dtype=np.float32))
        ok, diff = same(G.capture(env), ep["states"][str(d * PLAN)])
        row = dict(decision=d, cand=i, start_ok=ok, start_maxdiff=diff)
        if ok:
            for t in range(PLAN):
                env.get_obs(); env.take_action(np.asarray(cands[d, i, t], dtype=np.float32))
                if env.eval_success:
                    break
            policy._probe_decision = d + 1          # continuation seeds aligned with the decision index
            acts_c = []
            orig = env.take_action

            def rec(action, action_type="qpos", _orig=orig):
                acts_c.append(np.asarray(action, dtype=np.float32).tobytes())
                return _orig(action, action_type=action_type)
            env.take_action = rec
            try:
                while not env.eval_success and env.take_action_cnt < env.step_lim:
                    obs = env.get_obs()
                    mi.eval(env, policy, obs)
            finally:
                del env.take_action
            row.update(success=bool(env.eval_success), steps=int(env.take_action_cnt), fast_obs=bool(a.fast_obs),
                       cont_actions=len(acts_c), cont_actions_sha=hashlib.sha256(b"".join(acts_c)).hexdigest()[:16])
        row["wall_s"] = round(time.time() - t0, 1)
        row["host"] = socket.gethostname()
        env.close_env()
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
