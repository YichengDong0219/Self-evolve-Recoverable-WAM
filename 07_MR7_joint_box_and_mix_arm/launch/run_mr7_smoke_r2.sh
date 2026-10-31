#!/bin/bash
# MR7 smoke re-run (Claude 2026-10-30): the 11 states whose left-arm plans were falsely rejected by the boxchk closure
# bug (fixed in takeover_fixes.py) + 140084 (list crashed in the flaky expert pre-check). New out dir; r1 kept.
L=/workspace/fact_local
exec 9> $L/bench/.mr7_smoke_r2.lock; flock -n 9 || exit 1
echo "$(date '+%F %T') start" >> $L/bench/SEL_mr7_smoke_r2.launch.log
cd $L/code
SIM_SELECT_VARIANTS="pp+seeds+rack+jbox+ns+boxchk+flip,pp+seeds+best+rack+jbox+ns+boxchk+flip" \
  /workspace/miniconda3/envs/RoboTwin/bin/python sim_select.py $L/bench/jobs_mr7_smoke_r2.jsonl \
  $L/bench/SEL_mr7_smoke_r2 3,3,3,4,4,4,6,6,6,7,7,7 8 1 >> $L/bench/SEL_mr7_smoke_r2.out 2>&1
echo "$(date '+%F %T') end rc=$?" >> $L/bench/SEL_mr7_smoke_r2.launch.log
