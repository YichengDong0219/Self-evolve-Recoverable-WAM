#!/bin/bash
# Line 1 (Claude 2026-10-31, user: expand base-policy recoveries 50 -> 100, hang failures only). MR7 variants and joint
# gates exactly as run_mr7_topup.sh; 8 candidates x 2 repeats. Job list bench/jobs_mr7_line1.jsonl (make_jobs_line1.py).
# usage: run_mr7_line1.sh <gpu list> <shard k1,k2,../n>     (one instance per machine; flock)
L=/workspace/fact_local
exec 9> $L/bench/.mr7_line1.lock; flock -n 9 || { echo "already running"; exit 1; }
echo "$(date '+%F %T') start gpus=$1 shard=$2" >> $L/bench/SEL_mr7_line1.launch.log
cd $L/code
SIM_SELECT_JOINT_GATES=1 SIM_SELECT_VARIANTS="pp+seeds+rack+jbox+ns+boxchk+flip,pp+seeds+best+rack+jbox+ns+boxchk+flip" /workspace/miniconda3/envs/RoboTwin/bin/python sim_select.py \
  $L/bench/jobs_mr7_line1.jsonl $L/bench/SEL_mr7_line1 $1 8 2 $2 >> $L/bench/SEL_mr7_line1.out 2>&1
echo "$(date '+%F %T') end rc=$?" >> $L/bench/SEL_mr7_line1.launch.log
