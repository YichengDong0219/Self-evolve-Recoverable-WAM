#!/bin/bash
# ATC G1 (Claude, 2026-11-21; user: "先做一些不用GPU/轻量GPU的探针实验"): the released FACT (fact-wam) on the
# 11-14 chunk bank, same script and settings as the R2 run (wm_value_feats.py, sigma = 1, 4 draws, seed 0).
# Amax only, one fully idle RTX 4090 (smi 4-8, no compute process, < 1 GB used); never the RTX PRO 6000.
# Refuses if the output exists, the data copy is missing, or another G1 holds the lock.
set -u
T=/workspace/fact_local/atc_probes_1121
W=$T/wm_value; OUT=$W/feats_chunks_FACT.npz; LOG=$T/logs/g1.txt
mkdir -p $T/logs
[ -e $OUT ] && { echo "refuse: $OUT exists"; exit 3; }
[ -e $W/DATA_COPIED ] || { echo "refuse: data not copied"; exit 4; }
exec 9>$T/logs/g1.lock; flock -n 9 || { echo "refuse: G1 already running"; exit 5; }
PY=/workspace/miniconda3/envs/fact/bin/python
CK=/workspace/FACT/models/fact-wam/transformer
export PATH=/workspace/miniconda3/envs/fact/bin:/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4
export PYTHONPATH=/workspace/FACT:/workspace/fact_sft_amax HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID WAN_MODEL_DIR=/workspace/FACT/models/Wan2.2-TI2V-5B-Diffusers
export ROBOTWIN_NORM_PATH=/workspace/FACT/models/fact-wam/norm_stats_delta.json
G=""; U=""
for i in 4 5 6 7 8; do
  name=$(nvidia-smi -i $i --query-gpu=name --format=csv,noheader)
  case "$name" in *4090*) ;; *) continue ;; esac
  used=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)
  uuid=$(nvidia-smi -i $i --query-gpu=uuid --format=csv,noheader)
  napps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
  if [ "$used" -lt 1000 ] && [ "$napps" -eq 0 ]; then G=$i; U=$uuid; break; fi
done
[ -z "$G" ] && { echo "$(date '+%F %T') no idle 4090, not started" >> $LOG; exit 6; }
echo "$(date '+%F %T') start smi $G $U ckpt $CK" >> $LOG
cd /workspace/FACT
CUDA_VISIBLE_DEVICES=$U ROBOTWIN_ROOT=$W/data $PY -W ignore $T/code/wm_value_feats.py --ckpt $CK --root $W/data \
  --out $OUT --draws 4 --batch 8 --seed 0 > $T/logs/g1_feats.log 2>&1
rc=$?
echo "$(date '+%F %T') rc $rc $(grep FEATS_DONE $T/logs/g1_feats.log | tail -1)" >> $LOG
echo $rc > $T/logs/g1.rc
