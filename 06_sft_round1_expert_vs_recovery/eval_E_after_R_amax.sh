#!/bin/bash
# E-final eval on Amax, rerouted (Claude, 2026-10-30): the pipeline's E eval waited for GPU 8 from 05:53 on, but GPU 8
# is used by another user. That waiter was stopped; this script waits for the pipeline's R eval to finish
# on GPUs 4-7, then evaluates mugE_nv's final transformer on GPUs 4-7 with the same protocol (eval_task_amax.sh,
# 120 seeds 160000-160119, tag mugE_nv, output ~/fact_local/eval_mug/mugE_nv/). Single instance (flock).
set -u
exec 9> ~/fact_sft_amax/.eval_E_after_R.lock; flock -n 9 || { echo "already running"; exit 1; }
D=~/fact_sft_amax; LC=~/fact_local/code; E=~/fact_local/eval_mug; LOG=$D/mug_pipeline_nv.txt
log() { echo "$(date '+%F %T') [eval_E_after_R] $*" >> $LOG; }
idle() { for g in "$@"; do
  [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .)" = "0" ] || return 1
  [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024 ] || return 1; done; }
log "waiting for 'eval mugR_nv finished'"
until grep -q "eval mugR_nv finished" $LOG; do sleep 120; done
until idle 4 5 6 7; do sleep 120; done
TP=$(ls -d $D/runs/mugE_nv/models/checkpoint_*_step_3194/transformer | head -1)
i=0; pids=""
for g in 4 5 6 7; do awk -v i=$i 'NR%4==i' $E/seeds_all.txt > $E/seeds_mugE_nv_amax_$g.txt; i=$((i+1)); done
log "eval mugE_nv on GPUs 4 5 6 7 ($TP)"
for g in 4 5 6 7; do
  bash $LC/eval_task_amax.sh $g mugE_nv $TP $E/seeds_mugE_nv_amax_$g.txt hanging_mug $E > $E/eval_mugE_nv_gpu$g.log 2>&1 &
  pids="$pids $!"
done
wait $pids
log "eval mugE_nv finished: $(ls $E/mugE_nv/*.json 2>/dev/null | wc -l) results"
