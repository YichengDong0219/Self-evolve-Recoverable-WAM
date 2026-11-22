#!/bin/bash
# PRO 6000 (Blackwell, smi 0-3) copy of start_server_amax.sh (Claude, 2026-11-22; user: "能否在Pro6000上也并行").
# Same probe server and flags; only the interpreter differs: fact_bw venv (fact env + torch 2.7.1+cu128, sm_120).
# usage: start_server_bw.sh <variant> <gpu> <port> [extra flags...]
set -u
V=$1; G=$2; P=$3; shift 3
D=/workspace/fact_local/runs_server/server_$V; mkdir -p $D
cd /workspace/FACT
B=/workspace/venvs/fact_bw
export PATH=$B/bin:/workspace/miniconda3/envs/fact/bin:/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
_U=$(nvidia-smi -i $G --query-gpu=uuid --format=csv,noheader 2>/dev/null)  # pin by UUID
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${_U:-$G} PYTHONPATH=/workspace/FACT:/workspace/fact_local/code
date +%s > $D/start_unix
setsid nohup $B/bin/python /workspace/fact_local/code/inference_server_probe.py \
  --host 127.0.0.1 --port $P --model_id models/Wan2.2-TI2V-5B-Diffusers --transformer_path models/fact-wam/transformer \
  --stats_path models/fact-wam/norm_stats_delta.json --device cuda:0 --dtype bf16 --text_len 512 --t5_len 64 \
  --text_encoder_device cuda:0 --action_chunk 48 --num_frames 5 --num_inference_steps 20 --guidance_scale 0.0 "$@" \
  > $D/server.log 2>&1 < /dev/null &
echo $! > $D/pid
echo started $V pid $(cat $D/pid)
