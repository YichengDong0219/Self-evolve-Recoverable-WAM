"""Multi-chunk unit measurement (Claude, 2026-11-22; PROTOCOL addendum 3; user: "测量一下多个chunk之间的差异，给出你认为
最合理的长度"). Runs in the RoboTwin env against a STANDARD probe server (pi0, count 1, fast mode), exactly like
continue_candidates.py (the gate continuations), plus a unit length K and recording:
for (decision d, candidate i): fresh scene, replay executed[:48d], verify the capture, execute candidate i (48 steps,
get_obs every step, stop on success), then pi0 continues to the end (step_lim). FACT sampling seeds of the next K-1
plans after the candidate are candidate-specific (continuation stream, repeat = i + 1); every later plan uses the
shared seeds (repeat = 0) like the gate. K = 1 is the gate continuation itself (cont_actions_sha is comparable with
the gate rows). K = 2 and K = 3 share their first two chunks.
Recorded per pair (npz): privileged state every 12 steps from the decision to the end (FEATS), the 3 camera views at
PRM sizes (prm_common.real_views) at every plan boundary +48j, 1 <= j <= img_k, and all continuation actions.
Recording only reads state (poses, functional points, gripper values, check_success), it never steps or renders.
Output: <out-dir>/seed<S>.unit<K>.jsonl (one row per pair; appends; a pair already present is skipped) and
<out-dir>/rec/seed<S>_d<d>_c<i>_u<K>.npz.
usage: python continue_record.py --episode <ep>.json --pairs d:i,... --port <port> --unit K --out-dir <dir>
"""
import argparse
import hashlib
import json
import socket
import time
from pathlib import Path

import numpy as np

import g0_driver as G  # noqa: E402
from exec_candidates import VIEWS, episode_info, same  # noqa: E402
from prm_common import real_views  # noqa: E402

from unit_common import FEATS  # noqa: E402

PLAN, EVERY = 48, 12


def feats(env, rel):
    m, r = env.mug, env.rack
    mp = m.get_pose()
    fp = np.asarray(m.get_functional_point(0)[:3], float)
    rp = np.asarray(r.get_pose().p, float); rf = np.asarray(r.get_functional_point(0)[:3], float)
    lee = np.asarray(env.robot.get_left_ee_pose(), float).reshape(-1)
    ree = np.asarray(env.robot.get_right_ee_pose(), float).reshape(-1)
    if lee.size != 7 or ree.size != 7:
        raise RuntimeError(f"unexpected ee pose size {lee.size} {ree.size}")
    v = np.concatenate([[rel, env.take_action_cnt], mp.p, mp.q, fp, (rp + rf) / 2, lee, ree,
                        [env.robot.left_gripper_val, env.robot.right_gripper_val, float(env.eval_success),
                         float(env.check_success())]]).astype(float)
    assert v.size == len(FEATS)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=Path, required=True); ap.add_argument("--pairs", required=True)
    ap.add_argument("--port", type=int, required=True); ap.add_argument("--unit", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--img-k", type=int, default=6)
    a = ap.parse_args()
    K = a.unit
    (a.out_dir / "rec").mkdir(parents=True, exist_ok=True)
    ep = json.loads(a.episode.read_text())
    seed = ep["env_seed"]
    out = a.out_dir / f"seed{seed}.unit{K}.jsonl"
    done = set()
    if out.exists():
        done = {(r["decision"], r["cand"]) for r in map(json.loads, out.read_text().splitlines()) if r}
    z = np.load(a.episode.with_suffix(".tts.npz"))
    cands = z["cands"]
    args = G.build_args(ep["task"], ep["config"])
    env = G.task_env(ep["task"])
    info = episode_info(env, args, ep)
    acts = ep["executed_actions"]
    mi, policy = None, None                    # ONE client per process (single-connection FACT server)
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
        b0 = policy._probe_binding                      # (start_id, root_seed, repeat, count)
        for s in range(d * PLAN):
            env.take_action(np.asarray(acts[s], dtype=np.float32))
        ok, diff = same(G.capture(env), ep["states"][str(d * PLAN)])
        row = dict(decision=d, cand=i, unit=K, start_ok=ok, start_maxdiff=diff)
        if ok:
            base = d * PLAN
            caps, imgs, img_rel = [feats(env, 0)], [], []
            for t in range(PLAN):
                env.get_obs(); env.take_action(np.asarray(cands[d, i, t], dtype=np.float32))
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
                while not env.eval_success and env.take_action_cnt < env.step_lim:
                    rel = env.take_action_cnt - base
                    policy._probe_binding = (b0[0], b0[1], (i + 1) if rel < K * PLAN else 0, b0[3])
                    obs = env.get_obs()
                    if rel % PLAN == 0 and 0 < rel <= PLAN * a.img_k:
                        imgs.append(real_views(*[np.asarray(obs["observation"][v]["rgb"][..., :3], dtype=np.uint8)
                                                 for v in VIEWS]))
                        img_rel.append(rel)
                    mi.eval(env, policy, obs)
            finally:
                del env.take_action
            rel = env.take_action_cnt - base
            if caps[-1][0] != rel:
                caps.append(feats(env, rel))
            row.update(success=bool(env.eval_success), steps=int(env.take_action_cnt),
                       success_rel=int(rel) if env.eval_success else None,
                       cont_actions=len(acts_c), cont_actions_sha=hashlib.sha256(b"".join(acts_c)).hexdigest()[:16],
                       n_caps=len(caps), img_rel=img_rel)
            act = (np.frombuffer(b"".join(acts_c), dtype=np.float32).reshape(len(acts_c), -1) if acts_c
                   else np.zeros((0, 14), np.float32))
            np.savez_compressed(a.out_dir / "rec" / f"seed{seed}_d{d}_c{i}_u{K}.npz", caps=np.asarray(caps),
                                img_rel=np.asarray(img_rel, int),
                                head=np.asarray([x[0] for x in imgs], np.uint8).reshape(-1, 192, 256, 3),
                                left=np.asarray([x[1] for x in imgs], np.uint8).reshape(-1, 96, 128, 3),
                                right=np.asarray([x[2] for x in imgs], np.uint8).reshape(-1, 96, 128, 3), actions=act)
        row["wall_s"] = round(time.time() - t0, 1)
        row["host"] = socket.gethostname()
        env.close_env()
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
