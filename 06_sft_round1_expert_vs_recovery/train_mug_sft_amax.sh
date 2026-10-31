#!/bin/bash
# Amax copy of train_mug_sft.sh (Claude, 2026-10-29): /newhome paths, ~/fact_sft_amax, GPUs via FACT_GPU_IDS (default 4,5,6,7;
# nvidia-smi indices = CUDA indices under PCI_BUS_ID order while all 9 cards are up; checked before launch).
# usage: train_mug_sft.sh <arm_tag> <robotwin_root> <max_steps> [dataset_multipliers_json]
# R arm: multipliers {"*": 1.0, "Clean/hanging_mug_failprefix": 0.5} keeps the failure prefix at ~1/3 of samples.
# Starts from the released FACT transformer; 4 GPUs, ZeRO-2 (optimizer on GPU; FACT_DS_OFFLOAD=1 moves it to CPU,
# ~1.7x slower), effective batch 2x4x4 = 32. Smoke 10-28: ~19 s/step, peak 41 GB/GPU.
# Output: ~/fact_sft_amax/runs/<arm_tag>/ (checkpoints every 500 steps, wandb offline logs).
set -u
DEF='{"*": 1.0}'; TAG=$1; ROOT=$2; STEPS=$3; MULT=${4:-$DEF}
D=~/fact_sft_amax/runs/$TAG
[ -e $D/train.log ] && { echo "run dir exists: $D"; exit 2; }
mkdir -p $D
cd /workspace/FACT
export PATH=/workspace/miniconda3/envs/fact/bin:/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline WANDB_DIR=$D/wandb PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/FACT:/workspace/fact_sft_amax CUDA_DEVICE_ORDER=PCI_BUS_ID
export ROBOTWIN_ROOT=$ROOT FACT_PROJECT_DIR=$D FACT_MAX_STEPS=$STEPS G=${FACT_GPU_IDS:-4,5,6,7}; export FACT_GPU_IDS=$G FACT_DATASET_MULTIPLIERS="$MULT" FACT_CKPT_INTERVAL=${FACT_CKPT_INTERVAL:-500}
date +%s > $D/start_unix
python -c "from fact_train import launch_from_config; launch_from_config('fact_sft_configs.mug_sft_mask.config')" > $D/train.log 2>&1
rc=$?; date +%s > $D/stop_unix; echo "{\"name\":\"train_$TAG\",\"gpu\":\"$G\",\"rc\":$rc,\"wall_s\":$(( $(cat $D/stop_unix)-$(cat $D/start_unix) ))}" >> ~/fact_sft_amax/ledger.jsonl
echo rc=$rc
