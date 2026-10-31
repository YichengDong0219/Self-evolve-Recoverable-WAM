#!/bin/bash
# MR7 smoke, local 3090 share (Claude 2026-10-30, user allowed the 3090 box): 18 of the 40 states, pre-claimed on the
# cluster so neither side runs them twice. GPUs 3,4,6,7 (idle at launch), 3 workers each.
L=/workspace/fact_local
exec 9> $L/bench/.mr7_smoke_3090.lock; flock -n 9 || exit 1
echo "$(date '+%F %T') start" >> $L/bench/SEL_mr7_smoke_3090.launch.log
cd $L/code
SIM_SELECT_VARIANTS="pp+seeds+rack+jbox+ns+boxchk+flip,pp+seeds+best+rack+jbox+ns+boxchk+flip" \
  /workspace/miniconda3/envs/RoboTwin/bin/python sim_select.py $L/bench/jobs_mr7_smoke_3090.jsonl \
  $L/bench/SEL_mr7_smoke 3,3,3,4,4,4,6,6,6,7,7,7 8 1 >> $L/bench/SEL_mr7_smoke_3090.out 2>&1
echo "$(date '+%F %T') end rc=$?" >> $L/bench/SEL_mr7_smoke_3090.launch.log
