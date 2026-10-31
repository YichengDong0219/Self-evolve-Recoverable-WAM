#!/bin/bash
# usage: run_imagine.sh <sim_gpu> <port> <rollout_name>  -> runs/imagine/<name>.real_vs_imagined.mp4
set -u
G=$1; P=$2; N=$3; R=~/fact_g0_claude/runs; mkdir -p $R/imagine
cd /workspace_cluster/RoboTwin-FACT
export PATH=/workspace_cluster/miniconda3/envs/robotwin/bin:/workspace_cluster/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_EXTENSIONS_DIR=/workspace_cluster/.cache/openwam-robotwin-torch-extensions
export CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 FACT_PROBE_DIR=/workspace_cluster/fact_g0_claude/code
export PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/RoboTwin-FACT:/workspace_cluster/fact_g0_claude/code
t0=$(date +%s)
timeout 3000 /workspace_cluster/miniconda3/envs/robotwin/bin/python /workspace_cluster/fact_g0_claude/code/imagine_episode.py --rollout $R/$N.json --port $P --out $R/imagine/$N.real_vs_imagined.mp4 > $R/imagine/$N.log 2>&1
rc=$?; echo "{\"name\":\"imagine_$N\",\"gpu\":\"$G\",\"rc\":$rc,\"wall_s\":$(( $(date +%s)-t0 ))}" >> $R/ledger.jsonl
grep -o '{"video.*' $R/imagine/$N.log; echo rc=$rc
