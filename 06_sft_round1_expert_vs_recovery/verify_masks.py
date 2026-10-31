"""Verify action-loss mask / value penalty on the REAL training data through FACT's dataset + transform (Claude, 10-29).
For every episode of every dataset: frames 0, m-1, m, last (recovery: m = action_masked_until_frame) or
0, onset-1, onset, last (prefix), plus 3 random frames; compare with the expected masks. CPU only."""
import copy, importlib, json, random, sys
import numpy as np
mod = importlib.import_module(sys.argv[1])
dc = copy.deepcopy(mod.config["dataloaders"]["train"])
from fact_datasets import load_dataset
from fact_train import build_transform
import world_action_model.transformers  # noqa: F401
ds = load_dataset(dc["data_or_config"]); ds.set_transform(build_transform(dc["transform"]))
len(ds)
off = 0; bad = 0; n = 0; rng = random.Random(0); summary = {}
for sub in ds.datasets:
    name = str(sub.data_path).split("/")[-1]
    rows = sub.failure_rows_by_episode
    for k, (st, sp) in enumerate(zip(sub.episode_starts, sub.episode_stops)):
        ep = int(sub.episode_rows[k]["episode_index"]); L = int(sp - st); row = rows.get(ep)
        cut = None
        if row and "action_masked_until_frame" in row: cut = int(row["action_masked_until_frame"])
        if row and "failure_active_from_frame" in row: cut = int(row["failure_active_from_frame"])
        frames = {0, L - 1} | ({cut - 1, cut} if cut else set()) | {rng.randrange(L) for _ in range(3)}
        for f in sorted(x for x in frames if 0 <= x < L):
            d = ds[off + int(st) + f]
            alm, fa = float(d["action_loss_mask"]), float(d["failure_active_mask"])
            if row is None: exp = (1.0, 0.0)
            elif row.get("failure_episode"): exp = (0.0, 1.0 if f >= cut else 0.0)
            else: exp = (0.0 if f < cut else 1.0, 0.0)
            n += 1; ok = (alm, fa) == exp; bad += not ok
            s = summary.setdefault(name, {"checked": 0, "mismatch": 0}); s["checked"] += 1; s["mismatch"] += not ok
            if not ok and bad < 10: print("MISMATCH", name, ep, f, cut, (alm, fa), exp)
    off += len(sub)
print(json.dumps(summary)); print("total checked", n, "mismatches", bad)
