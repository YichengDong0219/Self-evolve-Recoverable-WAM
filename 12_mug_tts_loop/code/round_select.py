"""Which episodes of a TTS round enter the analysis (Claude, 2026-11-21; PROTOCOL addendum 2-3). Pure Python.
  - scan seeds [start, start + n_scan) in ascending order; a seed is VALID if <ep>.json exists (rollout finished),
    INVALID if <ep>.invalid exists (expert check failed), otherwise UNRESOLVED (rollout crashed / not run yet);
  - keep the first `keep` valid seeds; an unresolved seed before the cut-off makes the set undefined -> error
    (never silently skipped);
  - held-out gate scenes = kept seeds (ascending) [::5].
usage (report): python3 round_select.py <round dir> <scan start> [n_scan 140] [keep 100]
"""
import json
import sys
from pathlib import Path


def ep_path(rd, seed):
    return Path(rd) / f"hanging_mug_seed{seed}.json"


def scan(rd, start, n_scan=140):
    valid, invalid, unresolved = [], [], []
    for s in range(start, start + n_scan):
        p = ep_path(rd, s)
        if p.exists():
            valid.append(s)
        elif p.with_suffix(".invalid").exists():
            invalid.append(s)
        else:
            unresolved.append(s)
    return valid, invalid, unresolved


def kept_seeds(rd, start, n_scan=140, keep=100, strict=True):
    valid, invalid, unresolved = scan(rd, start, n_scan)
    if len(valid) < keep:
        if strict:
            raise RuntimeError(f"only {len(valid)} valid seeds (< {keep}); unresolved: {unresolved}")
        return valid, dict(valid=len(valid), invalid=invalid, unresolved=unresolved, cutoff=None)
    cut = valid[keep - 1]
    before = [s for s in unresolved if s < cut]
    if before and strict:
        raise RuntimeError(f"unresolved seeds before the cut-off {cut}: {before}")
    return valid[:keep], dict(valid=len(valid), invalid=[s for s in invalid if s < cut], unresolved=before, cutoff=cut)


def held_out(kept):
    return sorted(kept)[::5]


if __name__ == "__main__":
    rd, start = Path(sys.argv[1]), int(sys.argv[2])
    n_scan = int(sys.argv[3]) if len(sys.argv) > 3 else 140
    keep = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    valid, invalid, unresolved = scan(rd, start, n_scan)
    print(json.dumps(dict(valid=len(valid), invalid=len(invalid), unresolved=unresolved)))
    try:
        k, info = kept_seeds(rd, start, n_scan, keep)
        print(json.dumps(dict(kept=len(k), cutoff=info["cutoff"], held_out=held_out(k))))
    except RuntimeError as e:
        print("NOT FINAL:", e)
