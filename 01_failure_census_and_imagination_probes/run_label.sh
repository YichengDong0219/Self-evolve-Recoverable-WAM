#!/bin/bash
# usage: run_label.sh <gpu> <rollout_name>   -> runs/labels/<rollout_name>.label.json
set -u
G=$1; N=$2; R=~/fact_g0_claude/runs; mkdir -p $R/labels
[ -e $R/labels/$N.label.json ] && { echo "exists $N"; exit 0; }
cd /workspace_cluster/RoboTwin-FACT
export PATH=/workspace_cluster/miniconda3/envs/robotwin/bin:/workspace_cluster/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_EXTENSIONS_DIR=/workspace_cluster/.cache/openwam-robotwin-torch-extensions
export CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 FACT_PROBE_DIR=/workspace_cluster/fact_g0_claude/code
export PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/RoboTwin-FACT:/workspace_cluster/fact_g0_claude/code
t0=$(date +%s)
timeout 1500 /workspace_cluster/miniconda3/envs/robotwin/bin/python /workspace_cluster/fact_g0_claude/code/p1_label.py --rollout $R/$N.json --out $R/labels/$N.label.json > $R/labels/$N.log 2>&1
rc=$?; echo "{\"name\":\"label_$N\",\"gpu\":\"$G\",\"rc\":$rc,\"wall_s\":$(( $(date +%s)-t0 ))}" >> $R/ledger.jsonl
tail -1 $R/labels/$N.log | cut -c1-300; echo rc=$rc
