#!/bin/bash
# usage: run_takeover.sh <gpu> <rollout_name> <strategy> [branch_step]  -> ~/fact_takeover/smoke/<name>_<strategy>.{json,mp4,log}
set -u
G=$1; N=$2; ST=$3; B=${4:-336}; O=~/fact_takeover/smoke; mkdir -p $O
OUT=$O/${N}_step${B}_takeover${ST}
[ -e $OUT.json ] && { echo "exists"; exit 0; }
cd /workspace_cluster/RoboTwin-FACT
export PATH=/workspace_cluster/miniconda3/envs/robotwin/bin:/workspace_cluster/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_EXTENSIONS_DIR=/workspace_cluster/.cache/openwam-robotwin-torch-extensions
export CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 FACT_PROBE_DIR=/workspace_cluster/fact_g0_claude/code
export PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/RoboTwin-FACT:/workspace_cluster/fact_g0_claude/code
t0=$(date +%s)
timeout 900 /workspace_cluster/miniconda3/envs/robotwin/bin/python /workspace_cluster/fact_g0_claude/code/takeover_smoke.py \
  --rollout /workspace_cluster/fact_g0_claude/runs/$N.json --branch-step $B --strategy $ST --out $OUT > $OUT.log 2>&1
rc=$?; echo "{\"name\":\"$(basename $OUT)\",\"gpu\":\"$G\",\"rc\":$rc,\"wall_s\":$(( $(date +%s)-t0 ))}" >> $O/../ledger.jsonl
grep -o '{"env_seed.*' $OUT.log | cut -c1-400; echo "rc=$rc"
