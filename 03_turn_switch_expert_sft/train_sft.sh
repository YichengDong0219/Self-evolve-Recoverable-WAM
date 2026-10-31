#!/bin/bash
# FACT SFT on a converted turn_switch task dir (Claude, 2026-10-28).
# usage: train_sft.sh <arm_tag> <robotwin_root> [max_steps=2000]
# Starts from the released FACT transformer; 4 GPUs, ZeRO-2 (optimizer on GPU; FACT_DS_OFFLOAD=1 moves it to CPU,
# ~1.7x slower), effective batch 2x4x4 = 32. Smoke 10-28: ~19 s/step, peak 41 GB/GPU.
# Output: ~/fact_sft/runs/<arm_tag>/ (checkpoints every 500 steps, wandb offline logs).
set -u
TAG=$1; ROOT=$2; STEPS=${3:-2000}
D=~/fact_sft/runs/$TAG
[ -e $D/train.log ] && { echo "run dir exists: $D"; exit 2; }
mkdir -p $D
cd /workspace_cluster/FACT
export PATH=/workspace_cluster/miniconda3/envs/fact/bin:/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline WANDB_DIR=$D/wandb PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/fact_sft
export ROBOTWIN_ROOT=$ROOT FACT_PROJECT_DIR=$D FACT_MAX_STEPS=$STEPS FACT_GPU_IDS=0,1,2,3
date +%s > $D/start_unix
python -c "from fact_train import launch_from_config; launch_from_config('fact_sft_configs.turn_switch_sft.config')" > $D/train.log 2>&1
rc=$?; date +%s > $D/stop_unix; echo "{\"name\":\"train_$TAG\",\"gpu\":\"0,1,2,3\",\"rc\":$rc,\"wall_s\":$(( $(cat $D/stop_unix)-$(cat $D/start_unix) ))}" >> ~/fact_sft/ledger.jsonl
echo rc=$rc
