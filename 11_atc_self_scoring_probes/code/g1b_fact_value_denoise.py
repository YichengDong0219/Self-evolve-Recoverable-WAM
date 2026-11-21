"""ATC G1b (Claude, 2026-11-21; user: "先做一些不用GPU/轻量GPU的探针实验"): the released FACT's value after its own
stage-2 denoising, conditioned on each candidate chunk -- the path FACT's best-of-N uses at inference
(scripts/inference_server.py: 20 steps, guidance 0, UniPC from the Wan2.2 scheduler config with flow shift 3.0,
action-only so no future images; best-of-N ranks by the value, lower is better).

Inputs come from the same chunk dataset as G1 (the training transform): reference = cached VAE latent of frame 0,
normalized state, T5 prompt, gt_action = the candidate chunk (normalized delta, as in training), pred_action slot =
zeros (as the pipeline's stage 2). K draws of the initial future-state / value noise are shared by all candidates and
drawn in the same order as wm_value_feats.py (G1), so the step-0 x0 estimate can be checked against G1's value.
No FACT repo change: the transformer and its eval-mode action-only forward are used as the pipeline uses them.
usage: python g1b_fact_value_denoise.py --ckpt <dir> --root <ROBOTWIN_ROOT> --out <npz> [--draws 4] [--batch 8]
       [--steps 20] [--shift 3.0] [--seed 0] [--limit N]
  npz: side [N], value_final [N, K] (denoised, normalized; lower = better), value_step0 [N, K] (x0 at the first step)
"""
import argparse
import copy
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=4); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20); ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.environ["ROBOTWIN_ROOT"] = a.root
    cfg = importlib.import_module("fact_sft_configs.mouse_dpo").config
    from diffusers import UniPCMultistepScheduler
    from fact_datasets import load_dataset
    from fact_datasets.samplers.episode_mixture_sampler import EpisodeMixtureSampler
    from fact_train import utils
    from fact_train.transforms import build_transform
    from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
    from world_action_model.trainer.wa_casual_trainer import build_teacher_forcing_per_token_timestep, process_transformer

    dev, dt = torch.device("cuda"), torch.bfloat16
    dl = cfg["dataloaders"]["train"]
    ds = load_dataset(dl["data_or_config"]); ds.set_transform(build_transform(dl["transform"]))
    child = getattr(ds, "datasets", [ds])[0]
    starts, _, _ = EpisodeMixtureSampler._load_episode_ranges(child)
    task = Path(child.data_path)
    eps = [json.loads(l) for l in (task / "dpo_episodes.jsonl").read_text().splitlines() if l.strip()]
    if a.limit:
        eps = eps[:a.limit]
    mc = cfg["models"]
    t = CasualWorldActionTransformer.from_pretrained(a.ckpt, torch_dtype=dt)
    t.reset_action_heads(int(mc.get("action_dim", 14)))
    t = process_transformer(t, dict(mc.get("transformer", {})))
    t.set_robot_adapter_rank(int(mc.get("robot_adapter_rank", 0)))
    missing, unexpected = t.load_state_dict(utils.load_state_dict(a.ckpt), strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint keys do not match: {len(missing)} missing, {len(unexpected)} unexpected")
    t.to(dev, dtype=dt).eval()
    base = UniPCMultistepScheduler.from_pretrained(os.environ["WAN_MODEL_DIR"], subfolder="scheduler")
    base.register_to_config(flow_shift=float(a.shift))

    N, K = len(eps), a.draws
    first = {k: ds[int(starts[0])][k] for k in ("visual_latents", "action", "future_state", "value_normalized")}
    gen = torch.Generator().manual_seed(a.seed)
    noise = []
    for _ in range(K):   # same draw order as wm_value_feats.py: v, a, f, val
        torch.randn(first["visual_latents"].shape, generator=gen); torch.randn(first["action"].shape, generator=gen)
        f = torch.randn(first["future_state"].shape, generator=gen)
        v = torch.randn(torch.as_tensor(first["value_normalized"]).reshape(1, 1).shape, generator=gen)
        noise.append(dict(f=f, val=v))
    v_final = np.zeros((N, K), np.float32); v_step0 = np.zeros((N, K), np.float32)
    for b0 in range(0, N, a.batch):
        idx = list(range(b0, min(N, b0 + a.batch)))
        S = [ds[int(starts[e["episode_index"]])] for e in (eps[i] for i in idx)]
        bs = len(S)
        lat = torch.stack([s["visual_latents"] for s in S]).to(dev, dt)
        act = torch.stack([s["action"] for s in S]).to(dev, dt)
        fs_shape = torch.stack([s["future_state"] for s in S]).shape
        state = torch.stack([s["state"] for s in S]).to(dev, dt)
        prompt = torch.stack([s["prompt_embeds"] for s in S]).to(dev, dt)
        ref = lat[:, :, :1]
        fpt = lat.shape[-1] * lat.shape[-2] // 4
        dummy = torch.zeros_like(act)
        for k in range(K):
            fs = noise[k]["f"].to(dev)[None].expand(fs_shape).to(dt).clone()
            val = noise[k]["val"].to(dev)[None].expand(bs, 1, 1).to(dt).clone()
            sf, sv = copy.deepcopy(base), copy.deepcopy(base)
            sf.set_timesteps(a.steps, device=dev); sv.set_timesteps(a.steps, device=dev)
            for i, tt in enumerate(sv.timesteps):
                noise_t = tt.reshape(1, 1).to(device=dev, dtype=dt).expand(bs, 1)
                tok_ts = build_teacher_forcing_per_token_timestep(
                    batch_size=bs, num_state_tokens=int(state.shape[1]), num_ref_tokens=fpt,
                    num_pred_action_tokens=int(dummy.shape[1]), num_gt_action_tokens=int(act.shape[1]),
                    num_future_state_tokens=int(fs.shape[1]), num_value_tokens=int(val.shape[1]), num_noisy_latent_tokens=0,
                    noise_t=noise_t, device=dev, dtype=dt)
                with torch.no_grad():
                    _, _, fs_pred, v_pred = t(ref_latents=ref, timestep=tok_ts, encoder_hidden_states=prompt, return_dict=False,
                                              pred_action=dummy, gt_action=act, state=state, future_state=fs, value=val,
                                              action_only=True)
                if i == 0:
                    sig = float(sv.sigmas[0])
                    v_step0[idx, k] = (val.float() - sig * v_pred.float()).flatten(1).mean(1).cpu().numpy()
                fs = sf.step(fs_pred, tt, fs, return_dict=False)[0]
                val = sv.step(v_pred, tt, val, return_dict=False)[0]
            v_final[idx, k] = val.float().flatten(1).mean(1).cpu().numpy()
        print(f"{idx[-1] + 1}/{N}", flush=True)
    np.savez_compressed(a.out, side=np.asarray([e["side"] for e in eps]), value_final=v_final, value_step0=v_step0,
                        steps=a.steps, shift=a.shift, sigma0=float(sv.sigmas[0]))
    print("G1B_DONE", dict(N=N, K=K, final_range=[float(v_final.min()), float(v_final.max())],
                           step0_range=[float(v_step0.min()), float(v_step0.max())]), flush=True)


if __name__ == "__main__":
    main()
