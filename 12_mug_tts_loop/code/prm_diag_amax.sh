#!/bin/bash
# PRM reproducibility diagnosis on Amax (Claude, 2026-11-22): start prm_service.py on a fully idle card with the given
# extra flags, wait until ready, run prm_determinism.py --tag <tag>, stop the service.
# usage: bash prm_diag_amax.sh <smi> <round dir> <tag> [extra prm_service flags...]
set -u
G=$1; RD=$2; TAG=$3; shift 3
M=/workspace/fact_local/mug_loop_1121; P=/workspace/prm
PY=/workspace/venvs/prm_dopamine/bin/python
L=$RD/prm_diag.log; S=$RD/prm_service_$TAG.log
log() { echo "$(date '+%F %T') $*" >> $L; }
[ -z "$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader)" ] && \
  [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024 ] || { log "$TAG: gpu $G not idle"; exit 2; }
cd $M/code
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 \
  setsid $PY prm_service.py --model $P/models/GRM-2.0-8B-Preview --port 8650 \
    --prompt-src $P/Robo-Dopamine/examples/inference.py --blank-goal $P/Robo-Dopamine/examples/blank_goal.png "$@" \
    > $S 2>&1 < /dev/null &
SPID=$!; sleep 2; SPG=$(ps -o pgid= -p $SPID | tr -d ' ')
log "$TAG: service pid $SPID flags: $*"
for i in $(seq 90); do
  grep -q PRM_SERVICE_READY $S && break
  ps -p $SPID > /dev/null || { log "$TAG: service died"; exit 1; }
  sleep 20
done
$PY prm_determinism.py --round-dir $RD --tag $TAG >> $L 2>&1 || log "$TAG: prm_determinism rc $?"
kill -- -$SPG 2>/dev/null; sleep 10; kill -9 -- -$SPG 2>/dev/null
log "$TAG: done, service stopped"
