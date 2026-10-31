# rf_hanging_mug_v1: reusable recoverable failure states (FACT policy, RoboTwin hanging_mug)

Built 2026-10-30 by `code/build_rf_assets.py` from the MR6 collection (10-28/29). Read-only and write-once:
never edit this directory. A new collection or planner gets a new asset version or a separate results directory.

## What a row is
`index.jsonl` has one line per **failure state**. A failure state is a FACT rollout that failed, plus the replan
boundary where cuRobo takes over. 845 states (841 unique seeds), seeds 100000–156142, `demo_clean`. All of them are
disjoint from the eval seeds 160000–160119.

| field | meaning |
|---|---|
| `state_id` | `hanging_mug/<env_seed>/step<takeover_step>`: key for every later result |
| `rollout` | hard link (in `rollouts/`) to the FACT rollout record: executed actions + boundary states; `rollout_sha256` |
| `takeover_step` | policy step where the expert takes over (`code/mug_takeover_step.py`: dropped / stalled rule) |
| `failure_type` | `dropped` (mug reached hook height and fell) or `stalled` (never lifted, motionless) |
| `mr6_tier` | outcome under the MR6 planner (below) |
| `mr6` | candidates (variant, arm, contact point), every attempt with its gate verdicts, attempt dir; for tier A the selected attempt with capture npz, video, onset, rest frame, max joint step, G7/G8 and per-joint min/max |
| `expert_demo` | same-seed stock expert demo (json/npz, status), present for 132 states (130 successful) |

Hard links (no extra disk; clean-ups of `bench/`, `stream_rollouts/` or `expert_mug/` cannot break the asset):
`rollouts/` (845 FACT rollouts), `recoveries/<seed>_step<N>.{json,mp4,npz}` (the 130 selected MR6 recoveries;
paths in `mr6.selected.asset_files`), `expert/<seed>.{json,npz}` (`expert_demo.asset_files`). Rejected attempts
stay in `bench/SEL_v7/` (`mr6.dir`), referenced but not linked. Because the files are hard links, making the asset
read-only also makes the original files read-only.

## Tiers (MR6 evidence; the planner, not the state, is what changes between iterations)
| tier | n | meaning | reuse |
|---|---|---|---|
| `A_recovered` | 130 | MR6 recovered and passed G1–G6 with a capture. G7 pass 120; **G8 (joint range) pass 61** (right arm 61/117, left arm 0/13) | known-recoverable: first targets for a new planner, and the pairing anchor (all 130 have a successful expert demo) |
| `B_feasible_unrecovered` | 250 | whole-chain-feasible candidates existed, but no attempt passed all gates (mean 10.2 attempts) | next priority |
| `C_no_feasible_chain` | 454 | MR6 found no feasible (arm, contact) chain | may open up with planner changes (e.g. gripper-roll alternatives); low priority |
| `U_untried` | 11 | listed but never finished (crash or not reached) | run when convenient |

"Recoverable" here means recoverable by MR6. With the joint-range fix of MR7, tier-A states that only had
out-of-range recoveries need to be re-proven. The rollouts and takeover steps do not depend on the planner.

## Reproducing a state (no model needed)
Replay is bit-exact (gate G0 verified on the 3090 box and the 4090 servers; re-checked 10-30 from this asset on
state `hanging_mug/100001/step384`: same mug pose, same feasible list):

    source /workspace/fact_local/env.sh <gpu>
    python /workspace/fact_local/code/takeover_bench.py --rollout <asset>/rollouts/<file> \
        --step <takeover_step> --plan MR6 --variant pp+seeds+rack --out <new_dir>/x.json [--mr-list | --mr-forced ...]

Job list for the simulate-and-select driver:

    cd /workspace/fact_local/code
    python rf_assets.py A_recovered ../bench/jobs_<iter>.jsonl <group>      # tiers comma-separated or 'all'
    python sim_select.py ../bench/jobs_<iter>.jsonl ../bench/SEL_<iter> <gpu,...>

Use a new `group`, jobs file and out dir per iteration. Results stay keyed by `state_id`, so iterations can be
compared state by state.

## Validity conditions
- Same RoboTwin-FACT checkout (`envs/hanging_mug.py` sha in `meta.json`), `demo_clean`, same assets. Replay uses
  only recorded actions, so no FACT server is needed.
- Code shas of the MR6 pipeline at build time are in `meta.json`. The rollouts and takeover steps stay valid even
  when the planner code changes.
- `meta.json` also holds the G8 reference range (frozen-list expert demos ∪ FACT prefixes, ±0.5 rad).
