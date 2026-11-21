"""ATC pilot patches (Claude, 2026-11-21; user approved the A1 pilot). Imported by fact_sft_configs.atc_value.
No FACT repo change: monkeypatches applied at import time, active only for datasets / runs that opt in.
  1 episode exclusion  a task dir with meta/atc_exclude_episodes.json (list of episode_index) never has those episodes
                       sampled by EpisodeMixtureSampler (the A1 test states stay out of training).
  2 value target       with env ATC_VALUE_TABLE = JSON {episode_index: raw value}, the transform's value_normalized is
                       replaced by that raw value in FACT's normalisation (value stats min -1, max 2 -> [-1, 1]);
                       a missing episode raises (fail loudly). raw = 1 - p, p = the candidate's continuation success rate.
Frame-0-only windows come from fact_sft_configs.root_only_patch (meta/root_only.json).
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from fact_datasets.samplers.episode_mixture_sampler import EpisodeMixtureSampler
from world_action_model.transformers.wa_transforms_lerobot import WATransformsLerobot

VALUE_MIN, VALUE_MAX = -1.0, 2.0   # FACT default value stats (norm_stats_delta.json: value min -1, max 2)

if not getattr(EpisodeMixtureSampler, "_atc_exclude_patch", False):
    _orig_ranges = EpisodeMixtureSampler._load_episode_ranges

    def _load_episode_ranges(dataset):
        starts, stops, size = _orig_ranges(dataset)
        meta = Path(dataset.data_path) / "meta"
        p = meta / "atc_exclude_episodes.json"
        if p.is_file():
            ex = {int(x) for x in json.loads(p.read_text())}
            eps = [json.loads(l) for l in (meta / "episodes.jsonl").read_text().splitlines() if l.strip()]
            eps = [e for e in eps if int(e.get("length", 0)) > 0]
            if len(eps) != len(starts):
                raise RuntimeError(f"atc_patch: {len(eps)} episodes vs {len(starts)} ranges")
            keep = np.asarray([int(e["episode_index"]) not in ex for e in eps])
            starts, stops = np.asarray(starts)[keep], np.asarray(stops)[keep]
            print(f"[atc_patch] {dataset.data_path}: kept {int(keep.sum())}/{len(keep)} episodes", flush=True)
        return starts, stops, size

    EpisodeMixtureSampler._load_episode_ranges = staticmethod(_load_episode_ranges)
    EpisodeMixtureSampler._atc_exclude_patch = True

if not getattr(WATransformsLerobot, "_atc_value_patch", False):
    _orig_call = WATransformsLerobot.__call__
    _TABLE = {}

    def __call__(self, data_dict):
        ep = int(data_dict["episode_index"]) if "episode_index" in data_dict else None
        out = _orig_call(self, data_dict)
        path = os.environ.get("ATC_VALUE_TABLE")
        if path:
            if not _TABLE:
                _TABLE.update({int(k): float(v) for k, v in json.loads(Path(path).read_text()).items()})
            if ep is None:
                raise KeyError("atc_patch: sample without episode_index")
            raw = _TABLE[ep]
            vn = float(np.clip((raw - VALUE_MIN) / (VALUE_MAX - VALUE_MIN) * 2.0 - 1.0, -1.0, 1.0))
            out["value_normalized"] = torch.full_like(out["value_normalized"], vn)
        return out

    WATransformsLerobot.__call__ = __call__
    WATransformsLerobot._atc_value_patch = True
