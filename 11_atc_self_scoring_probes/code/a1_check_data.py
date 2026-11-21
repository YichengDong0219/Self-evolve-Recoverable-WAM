"""A1 pre-launch data check (Claude, 2026-11-21): through the training config's own dataset + transform + patched sampler,
verify (1) only train-state episodes are sampled, (2) any frame maps to frame 0, (3) value_normalized == injected target."""
import importlib, json, os
from pathlib import Path
import numpy as np
T = "/workspace/fact_local/atc_probes_1121"
os.environ.update(ROBOTWIN_ROOT=f"{T}/wm_value/data", ATC_VALUE_TABLE=f"{T}/atc_value_table.json",
                  FACT_INIT_CHECKPOINT="/workspace/fact_sft_amax/ckpts_mouse_r100v2/mouseR100v2_nv_step1573")
cfg = importlib.import_module("fact_sft_configs.atc_value").config
from fact_datasets import load_dataset
from fact_train.transforms import build_transform
from fact_datasets.samplers.episode_mixture_sampler import EpisodeMixtureSampler
dl = cfg["dataloaders"]["train"]
ds = load_dataset(dl["data_or_config"]); ds.set_transform(build_transform(dl["transform"]))
child = getattr(ds, "datasets", [ds])[0]
task = Path(child.data_path)
starts, stops, _ = EpisodeMixtureSampler._load_episode_ranges(child)
rows = [json.loads(l) for l in (task / "chunk_rows.jsonl").read_text().splitlines() if l.strip()]
eps = [json.loads(l) for l in (task / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
excl = set(json.load(open(task / "meta" / "atc_exclude_episodes.json")))
kept = [int(e["episode_index"]) for e in eps if int(e["episode_index"]) not in excl]
assert len(kept) == len(starts) == 1208, (len(kept), len(starts))
assert all(rows[k]["split"] == "train" for k in kept)
table = {int(k): v for k, v in json.load(open(os.environ["ATC_VALUE_TABLE"])).items()}
rng = np.random.default_rng(0); bad = 0
for j in rng.choice(len(starts), 12, replace=False):
    ep = kept[j]; gi = int(starts[j]) + int(rng.integers(0, 49))
    s = ds[gi]; s0 = ds[int(starts[j])]
    vn = float(np.asarray(s["value_normalized"]).ravel()[0])
    exp = ((table[ep] + 1.0) / 3.0) * 2.0 - 1.0
    same0 = bool(np.allclose(np.asarray(s["future_state"]), np.asarray(s0["future_state"])))
    ok = abs(vn - exp) < 1e-4 and same0
    bad += not ok
    print(f"ep {ep} split {rows[ep]['split']} label {rows[ep]['label']:.2f} value_norm {vn:+.4f} expected {exp:+.4f} frame0-mapped {same0} {'OK' if ok else 'BAD'}")
print("CHECK", "PASS" if bad == 0 else f"FAIL {bad}")
