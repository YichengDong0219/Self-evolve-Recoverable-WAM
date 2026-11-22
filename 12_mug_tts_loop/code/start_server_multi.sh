#!/bin/bash
# Multi-connection FACT probe server (inference_server_multi.py), CPU threads capped (Claude, 2026-11-22).
# usage: start_server_multi.sh <variant> <gpu> <port> <fact|fact_bw> [extra flags...]
#   fact    = 4090 (miniconda fact env, torch 2.6.0+cu124);  fact_bw = PRO 6000 (venv, torch 2.7.1+cu128, sm_120)
set -u
V=$1; G=$2; P=$3; E=$4; shift 4
D=/workspace/fact_local/runs_server/server_$V; mkdir -p $D
cd /workspace/FACT
if [ "$E" = fact_bw ]; then B=/workspace/venvs/fact_bw; PY=$B/bin/python
  export PATH=$B/bin:/workspace/miniconda3/envs/fact/bin:/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0
else PY=/workspace/miniconda3/envs/fact/bin/python
  export PATH=/workspace/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export FACT_SERVER_THREADS=${FACT_SERVER_THREADS:-4} OMP_NUM_THREADS=${FACT_SERVER_THREADS:-4} MKL_NUM_THREADS=${FACT_SERVER_THREADS:-4} OPENBLAS_NUM_THREADS=${FACT_SERVER_THREADS:-4}
_U=$(nvidia-smi -i $G --query-gpu=uuid --format=csv,noheader 2>/dev/null)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${_U:-$G} PYTHONPATH=/workspace/FACT:/workspace/fact_local/code:/workspace/fact_local/mug_loop_1121/code
date +%s > $D/start_unix
setsid nohup $PY /workspace/fact_local/mug_loop_1121/code/inference_server_multi.py \
  --host 127.0.0.1 --port $P --model_id models/Wan2.2-TI2V-5B-Diffusers --transformer_path models/fact-wam/transformer \
  --stats_path models/fact-wam/norm_stats_delta.json --device cuda:0 --dtype bf16 --text_len 512 --t5_len 64 \
  --text_encoder_device cuda:0 --action_chunk 48 --num_frames 5 --num_inference_steps 20 --guidance_scale 0.0 "$@" \
  > $D/server.log 2>&1 < /dev/null &
echo $! > $D/pid
echo started $V pid $(cat $D/pid)
