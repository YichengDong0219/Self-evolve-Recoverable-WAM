#!/bin/bash
# ATC A1 post-training scoring (Claude, 2026-11-21): waits for the A1 training to finish, then scores the step 300 / 200 /
# 100 snapshots with the G1b 20-step value path (one fully idle Amax 4090 each, retried until one is free). Read-only
# w.r.t. training; each scoring run refuses if its output exists.
set -u
T=/workspace/fact_local/atc_probes_1121; K=/workspace/fact_sft_amax/ckpts_atc_a1; LOG=$T/logs/a1_post.txt
exec 9>$T/logs/a1_post.lock; flock -n 9 || { echo "a1_post lock held"; exit 1; }
log() { echo "$(date '+%F %T') $*" >> $LOG; }
log "waiting for training"
until [ -e $T/logs/a1_full.rc ]; do sleep 120; done
rc=$(cat $T/logs/a1_full.rc); log "training rc $rc"
[ "$rc" = 0 ] || { log "training failed; no scoring"; exit 1; }
for s in 300 200 100; do
  ck=$K/atcA1_step$s; [ -e $ck/diffusion_pytorch_model.bin ] || { log "missing $ck"; continue; }
  until [ -e $T/logs/g1b_atcA1_step$s.txt ] && grep -q " start smi" $T/logs/g1b_atcA1_step$s.txt; do
    bash $T/code/g1b_score_amax.sh g1b_atcA1_step$s $ck > /dev/null 2>&1 < /dev/null &
    sleep 90
    grep -q " start smi" $T/logs/g1b_atcA1_step$s.txt 2>/dev/null && break
    log "no idle 4090 for step $s yet; retry in 5 min"; sleep 300
  done
  log "scoring step $s launched"
done
for s in 300 200 100; do until [ -e $T/logs/g1b_atcA1_step$s.rc ]; do sleep 60; done; log "step $s scoring rc $(cat $T/logs/g1b_atcA1_step$s.rc)"; done
touch $T/logs/A1_POST_DONE; log "end"
