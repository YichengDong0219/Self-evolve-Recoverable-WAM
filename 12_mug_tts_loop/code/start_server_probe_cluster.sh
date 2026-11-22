#!/bin/bash
# Standard probe server (pi0 = released FACT, PROBE build) on the 4090 cluster (Claude, 2026-11-22): cluster copy of
# fact_local/code/start_server_amax.sh -- same build (inference_server_probe.py, md5 a7875f2d), same model files,
# same flags; only the paths differ.
# usage: start_server_probe_cluster.sh <variant> <smi index> <port> [extra flags...]
set -u
V=$1; G=$2; P=$3; shift 3
M=/workspace_cluster/fact_local/mug_loop_1121; D=$M/runs_server/server_$V; mkdir -p $D
cd /workspace_cluster/FACT
export PATH=/workspace_cluster/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
_U=$(nvidia-smi -i $G --query-gpu=uuid --format=csv,noheader 2>/dev/null)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${_U:-$G} PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/fact_local/code
date +%s > $D/start_unix
setsid nohup /workspace_cluster/miniconda3/envs/fact/bin/python /workspace_cluster/fact_local/code/inference_server_probe.py \
  --host 127.0.0.1 --port $P --model_id models/Wan2.2-TI2V-5B-Diffusers --transformer_path models/fact-wam/transformer \
  --stats_path models/fact-wam/norm_stats_delta.json --device cuda:0 --dtype bf16 --text_len 512 --t5_len 64 \
  --text_encoder_device cuda:0 --action_chunk 48 --num_frames 5 --num_inference_steps 20 --guidance_scale 0.0 "$@" \
  > $D/server.log 2>&1 < /dev/null &
echo $! > $D/pid
echo started $V pid $(cat $D/pid)
