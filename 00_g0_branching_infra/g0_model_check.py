"""FACT G0 model-side numeric checks against a running probe server (Claude, 2026-10-27).

Uses one saved real observation (.npz from g0_driver.py --save-request). Sends:
  same_seed_a / same_seed_b : N=1, identical seed twice          -> must be bitwise equal
  other_seed                : N=1, a different seed              -> must differ (sanity)
  pool8                     : N=8, explicit candidate seeds      -> candidate pool + raw values + selected index
  pool_member_k             : N=1 with the seed of pool candidate k -> compare with pool8 candidate k
Writes one new JSON (refuses overwrite). Server variant (cache / return_images) is recorded via --variant.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from world_action_model.sockets import RobotInferenceClient

SEED_A, SEED_B = 1111111111, 2222222222
POOL = [3000000000 + i for i in range(8)]
K = 3


def as_np(x):
    return x.detach().cpu().float().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    z = np.load(a.request)
    base = {k.replace("__", "."): z[k] for k in z.files if k != "instruction"}
    base["instruction"] = str(z["instruction"])
    client = RobotInferenceClient(host="127.0.0.1", port=a.port, timeout_ms=600000)
    assert client.ping()

    def ask(seeds):
        req = dict(base, best_of_N=len(seeds), probe_seeds=list(seeds))
        resp = client.inference(req)
        if resp.get("probe_seeds") != list(seeds):
            raise RuntimeError("seed receipt mismatch")
        act = as_np(resp["action"]).reshape(-1, 14)
        row = {"seeds": list(seeds), "action_sha256": hashlib.sha256(np.ascontiguousarray(act).tobytes()).hexdigest(),
               "action": act.tolist(), "selected_index": resp.get("selected_index")}
        if resp.get("values_per_sample") is not None:
            row["values_per_sample"] = as_np(resp["values_per_sample"]).reshape(-1).tolist()
        if resp.get("probe_candidate_actions") is not None:
            row["candidates"] = as_np(resp["probe_candidate_actions"]).reshape(len(seeds), -1, 14).tolist()
        return row

    r = {"same_seed_a": ask([SEED_A]), "same_seed_b": ask([SEED_A]), "other_seed": ask([SEED_B]),
         "pool8": ask(POOL), "pool_member_k": ask([POOL[K]])}
    A = lambda k: np.asarray(r[k]["action"])  # noqa: E731
    cand_k = np.asarray(r["pool8"]["candidates"][K])
    summary = {
        "variant": a.variant,
        "same_seed_bitwise_equal": r["same_seed_a"]["action_sha256"] == r["same_seed_b"]["action_sha256"],
        "same_seed_max_abs": float(np.abs(A("same_seed_a") - A("same_seed_b")).max()),
        "other_seed_max_abs": float(np.abs(A("same_seed_a") - A("other_seed")).max()),
        "pool_member_vs_alone_max_abs": float(np.abs(cand_k - A("pool_member_k")).max()),
        "pool_member_vs_alone_bitwise": bool(np.array_equal(cand_k.astype(np.float32), A("pool_member_k").astype(np.float32))),
        "pool_selected_index": r["pool8"]["selected_index"],
        "pool_values": r["pool8"].get("values_per_sample"),
        "pool_pairwise_max_abs_mean": float(np.mean([np.abs(np.asarray(r["pool8"]["candidates"][i]) -
                                                            np.asarray(r["pool8"]["candidates"][j])).max()
                                                     for i in range(8) for j in range(i + 1, 8)])),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"format": "fact-g0-model-check/1", "summary": summary, "requests": r}))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
