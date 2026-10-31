#!/bin/bash
# MR7 top-up (Claude 2026-10-30, user: mix MR6+MR7 recoveries, fill to 50 valid within the E-arm 100 seeds, then manual
# review). 54 E-arm seeds without a valid recovery; joint gates G7/G8 at selection; 8 candidates x 2 repeats.
# Sharded with the cluster: this machine takes shards 0,1 of 3.
L=/workspace/fact_local
exec 9> $L/bench/.mr7_topup.lock; flock -n 9 || exit 1
echo "$(date '+%F %T') start" >> $L/bench/SEL_mr7_topup.launch.log
cd $L/code
SIM_SELECT_JOINT_GATES=1 SIM_SELECT_VARIANTS="pp+seeds+rack+jbox+ns+boxchk+flip,pp+seeds+best+rack+jbox+ns+boxchk+flip" /workspace/miniconda3/envs/RoboTwin/bin/python sim_select.py   $L/bench/jobs_mr7_topup.jsonl $L/bench/SEL_mr7_topup 3,3,3,4,4,4,6,6,6,7,7,7 8 2 0,1/3 >> $L/bench/SEL_mr7_topup.out 2>&1
echo "$(date '+%F %T') end rc=$?" >> $L/bench/SEL_mr7_topup.launch.log
