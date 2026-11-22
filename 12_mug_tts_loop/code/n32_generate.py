"""N = 32 candidates at the round-0 snapshots (Claude, 2026-11-22; PROTOCOL addendum 4; user: "选部分轨迹，在每个
snapshot出发rollout32次，然后让现有的critic打分，看看能否得到比K=8更高的多样性").
Runs in the RoboTwin env against the TTS server (start_server_tts_amax.sh: all candidate actions + imagined keyframes).
Per decision d of the round-0 episode: fresh scene, replay executed[:48d] bit-exactly (capture checked), then ONE plan
request of a fresh ProbeClient (count 8, decision d; FACT's request holds only the current 3 frames, the state and the
instruction, no history) is fanned out into 4 server calls with the seeds of candidate ids 0-7, 8-15, 16-23, 24-31
(seed_for(root, start, 'candidate', d, candidate=i)). The scene is never stepped: take_action is a no-op during that
call. Check: ids 0-7 against the recorded round-0 candidates (tts.npz cands[d]).
Output per state: <out>/seed<S>_d<d>.n32.npz (cands [32,48,14] float32, imgs [32,3,4,H,W] uint8, cur_head/left/right,
state) and <out>/seed<S>_d<d>.cands.npz (cands only, for the continuation machines); one row per state in
<out>/gen_seed<S>.jsonl (start_ok, ids 0-7 equal to round 0, server seconds per call). Skips states already done.
usage: python n32_generate.py --episode r0/hanging_mug_seed<S>.json --decisions 0,1,2,3,4,5,6 --port <p> --out-dir <dir>
"""
import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np

import g0_driver as G  # noqa: E402
from exec_candidates import episode_info, same  # noqa: E402
from seed_plan import seed_for  # noqa: E402
from tts_driver import to_np, to_uint8_hwc  # noqa: E402

PLAN, N, B = 48, 32, 8
VIEWS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class FanOut:
    """Outer wrapper around the ProbeClient's RecordingTransport: the client's own request (ids 0-7) plus 3 more server
    calls with the seeds of ids 8-31; stores everything; returns the first response (its action is never executed)."""

    def __init__(self, inner, root, start):
        self.inner, self.root, self.start, self.rows = inner, root, start, []

    def inference(self, request):
        d = int(request["probe_decision"])
        if list(request["probe_seeds"]) != [seed_for(self.root, self.start, "candidate", d, candidate=i) for i in range(B)]:
            raise RuntimeError("client seeds differ from candidate ids 0-7")
        cands, imgs, secs, first = [], [], [], None
        for off in range(0, N, B):
            req = dict(request)
            if off:
                req["probe_seeds"] = [seed_for(self.root, self.start, "candidate", d, candidate=i) for i in range(off, off + B)]
            t0 = time.time()
            resp = self.inner.inference(req)          # RecordingTransport checks the seed receipt
            secs.append(round(time.time() - t0, 2))
            c, im = resp.get("probe_candidate_actions"), resp.get("probe_candidate_images")
            if c is None or im is None:
                raise RuntimeError("server did not return candidate actions/images (TTS server with --return_images?)")
            cands.append(to_np(c).astype(np.float32)); imgs.append(to_np(im).astype(np.uint8))
            first = first or resp
        self.rows.append(dict(decision=d, cands=np.concatenate(cands), imgs=np.concatenate(imgs), server_s=secs,
                              cur=[to_uint8_hwc(request[f"observation.images.{v}"]) for v in VIEWS],
                              state=np.asarray(request["observation.state"], dtype=np.float32).reshape(-1),
                              instruction=str(request["instruction"])))
        return first

    def __getattr__(self, name):
        return getattr(self.inner, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=Path, required=True); ap.add_argument("--decisions", required=True)
    ap.add_argument("--port", type=int, required=True); ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    ep = json.loads(a.episode.read_text())
    seed = ep["env_seed"]
    rec = np.load(a.episode.with_suffix(".tts.npz"))["cands"]
    out = a.out_dir / f"gen_seed{seed}.jsonl"
    done = {r["decision"] for r in map(json.loads, out.read_text().splitlines()) if r} if out.exists() else set()
    args = G.build_args(ep["task"], ep["config"])
    env = G.task_env(ep["task"])
    info = episode_info(env, args, ep)
    acts = ep["executed_actions"]
    mi, policy = None, None
    for d in [int(x) for x in a.decisions.split(",")]:
        if d in done:
            continue
        t0 = time.time()
        G.setup_episode(env, args, seed, info, forced=ep["instruction"])
        if policy is None:
            mi, policy = G.make_policy(a.port)
        if isinstance(policy.client, FanOut):
            policy.client = policy.client.inner
        mi.reset_model(policy)
        policy.bind_probe(ep["start_id"], ep["root_seed"], G.server_sink([]), count=B)
        policy._probe_decision = d
        fan = FanOut(policy.client, ep["root_seed"], ep["start_id"])
        policy.client = fan
        for s in range(d * PLAN):
            env.take_action(np.asarray(acts[s], dtype=np.float32))
        ok, diff = same(G.capture(env), ep["states"][str(d * PLAN)])
        row = dict(decision=d, start_ok=ok, start_maxdiff=diff)
        if ok:
            obs = env.get_obs()
            env.take_action = lambda action, action_type="qpos": None     # never step: only the plan request is wanted
            try:
                mi.eval(env, policy, obs)
            finally:
                del env.take_action
            if len(fan.rows) != 1:
                raise RuntimeError(f"expected one plan request, got {len(fan.rows)}")
            r = fan.rows[0]
            c = r["cands"]
            np.savez_compressed(a.out_dir / f"seed{seed}_d{d}.n32.npz", cands=c, imgs=r["imgs"], cur_head=r["cur"][0],
                                cur_left=r["cur"][1], cur_right=r["cur"][2], state=r["state"])
            np.savez_compressed(a.out_dir / f"seed{seed}_d{d}.cands.npz", cands=c)
            row.update(n=int(len(c)), ids07_equal=bool(np.array_equal(c[:B], rec[d].astype(np.float32))),
                       ids07_maxdiff=float(np.abs(c[:B] - rec[d]).max()), server_s=r["server_s"], instruction=r["instruction"])
        row["wall_s"] = round(time.time() - t0, 1)
        row["host"] = socket.gethostname()
        env.close_env()
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
