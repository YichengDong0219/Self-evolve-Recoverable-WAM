"""Sim steps per recorded frame: FACT rollout (one take_action per frame) vs demo cadence (save_freq) (Claude, 10-29).

usage: python cadence_probe.py <rollout.json> <out.json>
Replays the rollout's executed actions on the same scene and counts scene.step() calls inside each take_action,
plus the commanded joint step |a_t - s_t| and the realised step |s_{t+1} - s_t| per frame. No model.
"""
import json
import sys

import numpy as np

import g0_driver as G

src = json.load(open(sys.argv[1]))
args = G.build_args(src["task"], src["config"])
env = G.task_env(src["task"])
seed, info = G.expert_seed(env, args, src["env_seed"], 1)
G.setup_episode(env, args, seed, info, forced=src["instruction"])
cnt = {"n": 0}
orig = env.scene.step


def step(*a, **k):
    cnt["n"] += 1
    return orig(*a, **k)


env.scene.step = step
rows = []
for t, act in enumerate(src["executed_actions"]):
    s0 = np.asarray(env.get_obs()["joint_action"]["vector"], dtype=np.float64)
    cnt["n"] = 0
    env.take_action(np.asarray(act, dtype=np.float32))
    s1 = np.asarray(env.get_obs()["joint_action"]["vector"], dtype=np.float64)
    a = np.asarray(act, dtype=np.float64)
    arm = [i for i in range(14) if i not in (6, 13)]
    rows.append({"t": t, "sim_steps": cnt["n"], "cmd": float(np.abs(a - s0)[arm].max()),
                 "real": float(np.abs(s1 - s0)[arm].max())})
    if env.eval_success:
        break
json.dump({"env_seed": src["env_seed"], "save_freq": int(env.save_freq), "rows": rows}, open(sys.argv[2], "w"))
ss = np.array([r["sim_steps"] for r in rows])
print(src["env_seed"], "save_freq", env.save_freq, "sim steps/frame: median", np.median(ss), "p10/p90", np.percentile(ss, [10, 90]),
      "min/max", ss.min(), ss.max())
