"""Frozen-R2 features for an outcome (value) head (Claude, 2026-11-14; user: "在 R2 上冻结骨干训练价值头，暂时不跑对照").

R2 is a fixed feature extractor: for every DPO side episode (observation + one 48-step chunk as the executed action
G), one no-grad forward at flow time sigma = 1 (all noisy inputs pure noise, so the features depend only on the
observation and the action), K noise draws shared by all samples (common random numbers). A hook on
_split_robot_outputs takes the final hidden states (after norm_out, before the decoders) and keeps, per draw:
  fs     mean of the future-state tokens   (attend: observation + action + future-state/value tokens)
  val    mean of the value tokens          (what FACT's value decoder reads)
  img    mean of the future-image tokens   (attend: observation + action + future state + images: "imagined future")
plus FACT's own value prediction (x0 of the value token, the baseline that ranked chosen vs rejected at chance).
usage: python wm_value_feats.py --ckpt <R2 dir> --root <ROBOTWIN_ROOT> --out <npz> [--draws 4] [--batch 8] [--seed 0]
  npz: keys (episode order of dpo_episodes.jsonl): side (str), feat_fs/feat_val/feat_img [N, K, D] float16,
       value_pred [N, K] float32
"""
import argparse
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
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.environ["ROBOTWIN_ROOT"] = a.root
    cfg = importlib.import_module("fact_sft_configs.mouse_dpo").config
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
    mc = cfg["models"]
    t = CasualWorldActionTransformer.from_pretrained(a.ckpt, torch_dtype=dt)
    t.reset_action_heads(int(mc.get("action_dim", 14)))
    t = process_transformer(t, dict(mc.get("transformer", {})))
    t.set_robot_adapter_rank(int(mc.get("robot_adapter_rank", 0)))
    missing, unexpected = t.load_state_dict(utils.load_state_dict(a.ckpt), strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint keys do not match: {len(missing)} missing, {len(unexpected)} unexpected")
    t.to(dev, dtype=dt).eval()

    cap = {}
    orig_split = t._split_robot_outputs

    def hooked(hidden_states, num_state_tokens, num_ref_tokens, num_pred_action_tokens, num_gt_action_tokens,
               num_future_state_tokens, num_value_tokens):
        rs = num_state_tokens + num_ref_tokens
        gt_end = rs + num_pred_action_tokens + num_gt_action_tokens
        fs_end = gt_end + num_future_state_tokens
        v_end = fs_end + num_value_tokens
        h = hidden_states.float()
        cap["fs"] = h[:, gt_end:fs_end].mean(1)
        cap["val"] = h[:, fs_end:v_end].mean(1)
        cap["img"] = h[:, v_end:].mean(1)  # robot tokens end at v_end; the future-image tokens follow
        return orig_split(hidden_states=hidden_states, num_state_tokens=num_state_tokens, num_ref_tokens=num_ref_tokens,
                          num_pred_action_tokens=num_pred_action_tokens, num_gt_action_tokens=num_gt_action_tokens,
                          num_future_state_tokens=num_future_state_tokens, num_value_tokens=num_value_tokens)

    t._split_robot_outputs = hooked
    N, K = len(eps), a.draws
    feats = {k: None for k in ("fs", "val", "img")}
    vpred = np.zeros((N, K), np.float32)
    first = {k: ds[int(starts[0])][k] for k in ("visual_latents", "action", "future_state", "value_normalized")}
    gen = torch.Generator().manual_seed(a.seed)
    noise = [dict(v=torch.randn(first["visual_latents"].shape, generator=gen), a=torch.randn(first["action"].shape, generator=gen),
                  f=torch.randn(first["future_state"].shape, generator=gen),
                  val=torch.randn(torch.as_tensor(first["value_normalized"]).reshape(1, 1).shape, generator=gen)) for _ in range(K)]
    for b0 in range(0, N, a.batch):
        idx = list(range(b0, min(N, b0 + a.batch)))
        S = [ds[int(starts[e["episode_index"]])] for e in (eps[i] for i in idx)]
        bs = len(S)
        lat = torch.stack([s["visual_latents"] for s in S]).to(dev, dt)
        act = torch.stack([s["action"] for s in S]).to(dev, torch.float32)
        fs = torch.stack([s["future_state"] for s in S]).to(dev, torch.float32)
        state = torch.stack([s["state"] for s in S]).to(dev, torch.float32)
        prompt = torch.stack([s["prompt_embeds"] for s in S]).to(dev, dt)
        val = torch.stack([torch.as_tensor(s["value_normalized"]).reshape(1, 1) for s in S]).to(dev, torch.float32)
        T, h, w = lat.shape[2], lat.shape[-2], lat.shape[-1]
        ffm = torch.ones(bs, 1, T, h, w, dtype=lat.dtype, device=dev); ffm[:, :, 0] = 0
        ts = torch.full((bs,), 1000, dtype=torch.long, device=dev)
        temp_ts = (ffm[:, :, :, ::2, ::2] * ts[:, None, None, None, None]).reshape(bs, -1)
        fpt = w * h // 4
        tok_ts = build_teacher_forcing_per_token_timestep(
            batch_size=bs, num_state_tokens=int(state.shape[1]), num_ref_tokens=fpt, num_pred_action_tokens=int(act.shape[1]),
            num_gt_action_tokens=int(act.shape[1]), num_future_state_tokens=int(fs.shape[1]), num_value_tokens=1,
            num_noisy_latent_tokens=fpt * (T - 1), noise_t=temp_ts[:, -2:-1], device=dev, dtype=dt)
        for k in range(K):
            nz = noise[k]
            n_v = nz["v"].to(dev, dt)[None].expand_as(lat)
            noisy = lat[:, :, :1] + n_v                     # sigma = 1: pure noise (+ the reference frame, as in training)
            ins = ((1 - ffm) * lat[:, :, :1] + ffm * noisy).to(dt)
            n_a = nz["a"].to(dev)[None].expand_as(act); n_f = nz["f"].to(dev)[None].expand_as(fs)
            n_val = nz["val"].to(dev)[None].expand_as(val)
            with torch.no_grad():
                _, _, _, vp = t._forward_train(
                    ref_latents=ins[:, :, :1], noisy_latents=ins[:, :, 1:], timestep=tok_ts, encoder_hidden_states=prompt,
                    return_dict=False, action=n_a.to(dt), gt_action=act.to(dt), state=state.to(dt), future_state=n_f.to(dt),
                    value=n_val.to(dt))
            vpred[idx, k] = (n_val.float() - vp.float()).flatten(1).mean(1).cpu().numpy()
            for key in feats:
                x = cap[key].cpu().numpy().astype(np.float16)
                if feats[key] is None:
                    feats[key] = np.zeros((N, K, x.shape[1]), np.float16)
                feats[key][idx, k] = x
        print(f"{idx[-1] + 1}/{N}", flush=True)
    np.savez_compressed(a.out, side=np.asarray([e["side"] for e in eps]), value_pred=vpred,
                        **{f"feat_{k}": v for k, v in feats.items()})
    print("FEATS_DONE", {k: v.shape for k, v in feats.items()}, flush=True)


if __name__ == "__main__":
    main()
