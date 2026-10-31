#!/bin/bash
# MR7 takeover smoke test on Amax (Claude, 2026-10-30, user request). Waits until the R-arm training has stopped and
# Amax smi 6 and 7 are idle (no compute app, < 1 GB), then runs simulate-and-select with the MR7 variants on the
# 40 asset states of bench/jobs_mr7_smoke.jsonl (3 simulator workers per card). Single instance (flock).
# usage (on Amax): setsid nohup bash mr7_smoke_launch.sh > /dev/null 2>&1 < /dev/null &
set -u
L=/workspace/fact_local
LOG=$L/bench/SEL_mr7_smoke.launch.log
exec 9> $L/bench/.mr7_smoke.lock
flock -n 9 || { echo "$(date '+%F %T') another launcher is running" >> $LOG; exit 1; }
echo "$(date '+%F %T') waiting for training stop + idle smi 6,7" >> $LOG
while true; do
  ok=1
  [ -e /workspace/fact_sft_amax/runs/mugR_nv/stop_unix ] || ok=0
  for g in 6 7; do
    u=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
    m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$n" = 0 ] && [ "$m" -lt 1024 ] || ok=0
  done
  [ $ok = 1 ] && break
  sleep 60
done
echo "$(date '+%F %T') start" >> $LOG
cd $L/code
SIM_SELECT_VARIANTS="pp+seeds+rack+jbox+ns+boxchk+flip,pp+seeds+best+rack+jbox+ns+boxchk+flip" \
  /workspace/miniconda3/envs/RoboTwin/bin/python sim_select.py ../bench/jobs_mr7_smoke.jsonl \
  ../bench/SEL_mr7_smoke 6,6,6,7,7,7 8 1 >> $L/bench/SEL_mr7_smoke.out 2>&1
echo "$(date '+%F %T') end rc=$?" >> $LOG
