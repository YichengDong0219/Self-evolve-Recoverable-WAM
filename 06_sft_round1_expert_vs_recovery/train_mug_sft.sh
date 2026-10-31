#!/bin/bash
# FACT SFT for the hanging_mug E vs R comparison (Claude, 2026-10-29; copy of train_sft.sh with the mug config).
# usage: train_mug_sft.sh <arm_tag> <robotwin_root> <max_steps> [dataset_multipliers_json]
# R arm: multipliers {"*": 1.0, "Clean/hanging_mug_failprefix": 0.5} keeps the failure prefix at ~1/3 of samples.
# Starts from the released FACT transformer; 4 GPUs, ZeRO-2 (optimizer on GPU; FACT_DS_OFFLOAD=1 moves it to CPU,
# ~1.7x slower), effective batch 2x4x4 = 32. Smoke 10-28: ~19 s/step, peak 41 GB/GPU.
# Output: ~/fact_sft/runs/<arm_tag>/ (checkpoints every 500 steps, wandb offline logs).
set -u
DEF='{"*": 1.0}'; TAG=$1; ROOT=$2; STEPS=$3; MULT=${4:-$DEF}
D=~/fact_sft/runs/$TAG
[ -e $D/train.log ] && { echo "run dir exists: $D"; exit 2; }
mkdir -p $D
cd /workspace_cluster/FACT
export PATH=/workspace_cluster/miniconda3/envs/fact/bin:/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline WANDB_DIR=$D/wandb PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/fact_sft
export ROBOTWIN_ROOT=$ROOT FACT_PROJECT_DIR=$D FACT_MAX_STEPS=$STEPS FACT_GPU_IDS=0,1,2,3 FACT_DATASET_MULTIPLIERS="$MULT" FACT_CKPT_INTERVAL=${FACT_CKPT_INTERVAL:-500}
date +%s > $D/start_unix
python -c "from fact_train import launch_from_config; launch_from_config('fact_sft_configs.mug_sft_mask.config')" > $D/train.log 2>&1
rc=$?; date +%s > $D/stop_unix; echo "{\"name\":\"train_$TAG\",\"gpu\":\"0,1,2,3\",\"rc\":$rc,\"wall_s\":$(( $(cat $D/stop_unix)-$(cat $D/start_unix) ))}" >> ~/fact_sft/ledger.jsonl
echo rc=$rc
