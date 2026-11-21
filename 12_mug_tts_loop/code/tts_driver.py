"""Mug TTS rollout driver (Claude, 2026-11-21; user: "开始执行"). Protocol: 12_mug_tts_loop/PROTOCOL.md.
Runs in the RoboTwin env (cwd = RoboTwin-FACT, PYTHONPATH incl. fact_local/code) against the TTS server
(inference_server_tts.py, --return_images). One episode per invocation, explicit expert-valid env seed (G0 rules).

At every 48-step plan boundary the ProbeClient asks for N candidates with explicit seeds
(seed_for(root, start, 'candidate', decision, i)); the server returns all N action chunks + their imagined keyframes.
A wrapper then picks one candidate and writes it into response["action"], which the stock FACT client executes:
  --select random   k = seed_for(root, start, 'selection', decision) % N        (round 0 / control C)
  --select critic   k = argmax of the critic service's scores (ties: lowest index among the max after a seeded shuffle)
Recorded per decision: step, chosen k, FACT's own BoN index and values, current 3-view frames, all candidate actions,
all candidate imagined keyframes. Recorded per episode (G0 format): executed actions, state capture at every boundary.
Outputs: <out>.json (refuses to overwrite) and <out>.tts.npz (frames, candidates, imagined keyframes).
"""
import argparse
import hashlib
import json
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

import g0_driver as G  # noqa: E402  (sets sys.path for RoboTwin-FACT and imports the sim)
from seed_plan import seed_for  # noqa: E402

PLAN = 48


def to_uint8_hwc(chw01):
    a = np.asarray(chw01.detach().cpu().numpy() if hasattr(chw01, "detach") else chw01, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.max() <= 1.0 + 1e-6:
        a = a * 255.0
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class SelectingTransport:
    """Outer wrapper around the ProbeClient's RecordingTransport: choose a candidate, overwrite response['action']."""

    def __init__(self, inner, pick, store):
        self.inner, self.pick, self.store = inner, pick, store

    def inference(self, request):
        t0 = time.time()
        response = self.inner.inference(request)
        t_srv = time.time() - t0
        d = int(request["probe_decision"])
        cands = response.get("probe_candidate_actions")
        imgs = response.get("probe_candidate_images")
        if cands is None or imgs is None:
            raise RuntimeError("server did not return candidate actions/images (TTS server with --return_images?)")
        cur = [to_uint8_hwc(request[f"observation.images.{v}"]) for v in ("cam_high", "cam_left_wrist", "cam_right_wrist")]
        state = np.asarray(request["observation.state"], dtype=np.float32).reshape(-1)
        cand_np, imgs_np = to_np(cands).astype(np.float32), to_np(imgs).astype(np.uint8)
        t1 = time.time()
        k, scores = self.pick(d, cur, state, cand_np, imgs_np, str(request["instruction"]))
        t_pick = time.time() - t1
        response["action"] = cands[k]
        vps = response.get("values_per_sample")
        self.store.append(dict(decision=d, k=int(k), scores=None if scores is None else [float(s) for s in scores],
                               fact_bon_index=int(response.get("selected_index", -1)),
                               fact_values=None if vps is None else [float(v) for v in to_np(vps).reshape(-1)],
                               server_s=round(t_srv, 2), pick_s=round(t_pick, 2),
                               cur=cur, state=state, cands=cand_np, imgs=imgs_np))
        return response

    def __getattr__(self, name):
        return getattr(self.inner, name)


def make_picker(mode, root_seed, start_id, n, critic_addr, start_frames):
    if mode == "random":
        def pick(d, cur, state, cands, imgs, instr):
            return seed_for(root_seed, start_id, "selection", d) % n, None
        return pick
    host, port = critic_addr.split(":")
    conn = Client((host, int(port)), authkey=b"mug-tts-critic")

    def pick(d, cur, state, cands, imgs, instr):
        if d == 0:
            start_frames[:] = cur
        conn.send(dict(op="score", instruction=instr, start=start_frames, cur=cur, state=state, cands=cands, imgs=imgs))
        rep = conn.recv()
        if "error" in rep:
            raise RuntimeError("critic: " + rep["error"])
        s = np.asarray(rep["scores"], dtype=np.float64)
        rng = np.random.default_rng(seed_for(root_seed, start_id, "selection", d))
        order = rng.permutation(n)
        k = int(order[np.argmax(s[order])])  # seeded tie-break
        return k, s.tolist()
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hanging_mug"); ap.add_argument("--config", default="demo_clean")
    ap.add_argument("--start-seed", type=int, required=True); ap.add_argument("--max-tries", type=int, default=1)
    ap.add_argument("--root-seed", type=int, default=20261121); ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--select", choices=["random", "critic"], required=True)
    ap.add_argument("--critic-addr", default="127.0.0.1:8650"); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    args = G.build_args(a.task, a.config)
    env = G.task_env(a.task)
    seed, info = G.expert_seed(env, args, a.start_seed, a.max_tries)
    start_id = f"{a.task}/{a.config}/{seed}"
    instruction = G.setup_episode(env, args, seed, info)
    mi, policy = G.make_policy(a.port)
    mi.reset_model(policy)
    io_log, store, start_frames = [], [], []
    policy.bind_probe(start_id, a.root_seed, G.server_sink(io_log), count=a.n)
    policy._probe_decision = 0
    policy.client = SelectingTransport(policy.client,
                                       make_picker(a.select, a.root_seed, start_id, a.n, a.critic_addr, start_frames), store)
    executed, states, t_ep = [], {}, time.time()
    while env.take_action_cnt < env.step_lim:
        step = int(env.take_action_cnt)
        if step % PLAN == 0:
            states[str(step)] = G.capture(env)
        obs = env.get_obs()
        before = len(executed)
        orig = env.take_action

        def rec(action, action_type="qpos", _orig=orig):
            executed.append(np.asarray(action, dtype=np.float32).tolist())
            return _orig(action, action_type=action_type)
        env.take_action = rec
        try:
            mi.eval(env, policy, obs)
        finally:
            del env.take_action
        if len(executed) != before + 1 or int(env.take_action_cnt) != step + 1:
            raise RuntimeError(f"step bookkeeping failed at step {step}")
        if env.eval_success:
            break
    states["final"] = G.capture(env)
    success, steps = bool(env.eval_success), int(env.take_action_cnt)
    env.close_env()
    for d, row in enumerate(store):
        if row["decision"] != d:
            raise RuntimeError("decision order mismatch")
        k = row["k"]
        if not np.allclose(np.asarray(executed[d * PLAN:(d + 1) * PLAN]), row["cands"][k][:len(executed[d * PLAN:(d + 1) * PLAN])], atol=1e-5):
            raise RuntimeError(f"executed actions differ from the chosen candidate at decision {d}")
    out = {"format": "fact-g0-episode/1+tts", "mode": "tts_" + a.select, "task": a.task, "config": a.config, "env_seed": seed,
           "instruction": instruction, "instruction_source": "generated", "root_seed": a.root_seed, "start_id": start_id,
           "branch_step": 0, "actions_from": None, "success": success, "steps": steps,
           "step_lim": int(env.step_lim), "expert_info": info, "executed_actions": executed,
           "executed_actions_sha256": hashlib.sha256(json.dumps(executed).encode()).hexdigest(), "states": states,
           "server_io": [{k: v for k, v in r.items() if k != "action"} for r in io_log],
           "tts": {"select": a.select, "n": a.n, "wall_s": round(time.time() - t_ep, 1),
                   "decisions": [{k: v for k, v in r.items() if k not in ("cur", "state", "cands", "imgs")} for r in store]}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out.with_suffix(".tts.npz"),
                        cur=np.stack([np.stack(r["cur"][:1]) for r in store])[:, 0],
                        cur_left=np.stack([r["cur"][1] for r in store]), cur_right=np.stack([r["cur"][2] for r in store]),
                        state=np.stack([r["state"] for r in store]), cands=np.stack([r["cands"] for r in store]),
                        imgs=np.stack([r["imgs"] for r in store]), k=np.asarray([r["k"] for r in store]))
    tmp = a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.rename(a.out)
    print(json.dumps({"env_seed": seed, "success": success, "steps": steps, "decisions": len(store),
                      "wall_s": out["tts"]["wall_s"], "server_s": [r["server_s"] for r in store]}))


if __name__ == "__main__":
    main()
