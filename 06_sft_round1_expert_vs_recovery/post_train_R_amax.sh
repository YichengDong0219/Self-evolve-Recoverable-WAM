#!/bin/bash
# Replaces the tail of mug_pipeline_amax_nv.sh after the user's 10-30 decision (evaluate up to step 3000 only; no
# step-3194 evals; Amax 4090s join the cluster eval pool via ckpt_eval_sched.py). Waits for the running R training
# (train_mug_sft_amax_nv.sh writes stop_unix and a ledger line), removes its DeepSpeed states, logs, and stops the
# pipeline's guard loop. Single instance (flock).
set -u
exec 9> ~/fact_sft_amax/.post_train_R.lock; flock -n 9 || { echo "already running"; exit 1; }
D=~/fact_sft_amax; R=$D/runs/mugR_nv; LOG=$D/mug_pipeline_nv.txt; GUARD=$1
log() { echo "$(date '+%F %T') [post_train_R] $*" >> $LOG; }
log "waiting for mugR_nv training to end (no step-3194 evals: user decision 10-30)"
until [ -e $R/stop_unix ]; do sleep 120; done
TP=$(ls -d $R/models/checkpoint_*_step_3194/transformer 2>/dev/null | head -1)
log "train mugR_nv finished: $(tail -1 $D/ledger.jsonl); final transformer ${TP:-MISSING}"
[ -n "$TP" ] && rm -rf $R/models/checkpoint_*/pytorch_model && log "mugR_nv DeepSpeed states removed"
kill $GUARD 2>/dev/null && log "guard loop stopped; pipeline finished (evals run by ckpt_eval_sched.py)"
