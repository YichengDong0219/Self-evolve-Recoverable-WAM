#!/bin/bash
# usage: run_cell.sh <sim_gpu> <port> <name> <source_json> <branch_step> <stream_repeat>  -> ~/fact_selfsample_R92/cells/<name>.json
set -u
G=$1; P=$2; N=$3; SRC=$4; B=$5; RP=$6
C=~/fact_selfsample_R92/cells; mkdir -p $C
[ -e $C/$N.json ] && { echo "exists $N"; exit 0; }
SEED=$(python3 -c "import json;print(json.load(open('$SRC'))['env_seed'])")
cd /workspace_cluster/RoboTwin-FACT
export PATH=/workspace_cluster/miniconda3/envs/robotwin/bin:/workspace_cluster/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_EXTENSIONS_DIR=/workspace_cluster/.cache/openwam-robotwin-torch-extensions
export CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 FACT_PROBE_DIR=/workspace_cluster/fact_g0_claude/code
export PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/RoboTwin-FACT:/workspace_cluster/fact_g0_claude/code
t0=$(date +%s)
timeout 1500 /workspace_cluster/miniconda3/envs/robotwin/bin/python /workspace_cluster/fact_g0_claude/code/branch_driver_codex.py branch \
  --task hanging_mug --config demo_clean --start-seed $SEED --max-tries 1 --port $P \
  --actions-from $SRC --branch-step $B --stream-repeat $RP --out $C/$N.json > $C/$N.log 2>&1
rc=$?; echo "{\"name\":\"$N\",\"gpu\":\"$G\",\"port\":$P,\"rc\":$rc,\"wall_s\":$(( $(date +%s)-t0 ))}" >> $C/../cells_ledger.jsonl
tail -1 $C/$N.log | cut -c1-250; echo rc=$rc
