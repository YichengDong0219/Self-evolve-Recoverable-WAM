"""Gate states for the critic gate (Claude, 2026-11-21; PROTOCOL step 5 + addendum 3). Needs numpy.
Held-out scenes = kept seeds [::5] (20 of the first 100 expert-valid, ascending; round_select.py). From all decision
states of the held-out episodes, listed in (seed, decision) order, draw --n-states (70) without replacement with
numpy default_rng(0). Every one of the 8 candidates of a drawn state is then continued with pi0
(continue_candidates.py). The draw happens once, before any continuation or critic result exists.
Output: <round dir>/gate_pairs.json (refuses to overwrite).
Re-test (PROTOCOL addendum 2, 11-22): --exclude <pairs json> --n-states 0 --out gate_pairs_retest.json lists EVERY
held-out decision state not in the excluded file (no random draw), in (seed, decision) order.
usage: python gate_select.py --round-dir <dir> --scan-start 170000 [--n-states 70] [--exclude f --out f]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from round_select import held_out, kept_seeds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True); ap.add_argument("--scan-start", type=int, required=True)
    ap.add_argument("--scan-n", type=int, default=140); ap.add_argument("--keep", type=int, default=100)
    ap.add_argument("--n-states", type=int, default=70, help="0 = all remaining states (no draw)")
    ap.add_argument("--n-cands", type=int, default=8)
    ap.add_argument("--exclude", type=Path, help="pairs json whose states are excluded")
    ap.add_argument("--out", default="gate_pairs.json")
    a = ap.parse_args()
    out = a.round_dir / a.out
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    kept, sel = kept_seeds(a.round_dir, a.scan_start, a.scan_n, a.keep)
    held = held_out(kept)
    states = []
    for s in held:
        k = np.load(a.round_dir / f"hanging_mug_seed{s}.tts.npz")["k"]
        states += [(s, d) for d in range(int(k.shape[0]))]
    n_all = len(states)
    if a.exclude:
        ex = {(x["seed"], x["decision"]) for x in json.loads(a.exclude.read_text())["states"]}
        states = [st for st in states if st not in ex]
    if a.n_states == 0:
        pick = sorted(states)
    else:
        idx = np.random.default_rng(0).choice(len(states), size=a.n_states, replace=False)
        pick = sorted(states[i] for i in idx)
    by_ep = {}
    for s, d in pick:
        by_ep.setdefault(str(s), []).append(d)
    out.write_text(json.dumps(dict(scan_start=a.scan_start, kept=len(kept), cutoff=sel["cutoff"], held_out=held,
                                   held_out_states=n_all, excluded_from=str(a.exclude) if a.exclude else None,
                                   n_states=len(pick), rng="numpy default_rng(0)" if a.n_states else "none (all)",
                                   states=[dict(seed=s, decision=d) for s, d in pick],
                                   pairs={s: ",".join(f"{d}:{i}" for d in ds for i in range(a.n_cands))
                                          for s, ds in by_ep.items()}), indent=1))
    print(json.dumps(dict(held_out=len(held), held_out_states=n_all, selected=len(pick), episodes=len(by_ep))))


if __name__ == "__main__":
    main()
