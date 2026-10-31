#!/bin/bash
# hanging_mug Mix arm (50 recovery + 50 expert, seeds = the E arm's 100) on Amax (Claude, 2026-10-30, user request).
# The E arm is NOT retrained: its mugE_nv checkpoints are the control. Everything below matches mugE_nv except data:
# config fact_sft_configs/mug_sft_novalue (value loss 0, penalty 0), released FACT transformer as init, 3194 steps,
# ckpt every 1000 + final, GPUs 4-7, eff. batch 32. Datasets (data/mugM/Clean):
#   hanging_mug_expert      50 stock expert demos, full loss (as in the E arm)
#   hanging_mug_recovery    50 curobo recoveries, action loss from curobo's first motion on (action_masked_until)
#   hanging_mug_failprefix  50 FACT failure prefixes (failure episodes, no action loss), multiplier 0.5
# Evaluation is not done here: snap_ckpts_mix.sh copies each checkpoint to ~/fact_sft_amax/ckpts/mugM_nv_step<N> and
# the local ckpt_eval_sched.py evaluates it on the first idle 4090 (Amax or the old cluster).
# usage: setsid nohup bash mug_pipeline_amax_mix.sh <list_dir> > ~/fact_sft_amax/mug_pipeline_mix.out 2>&1 < /dev/null &
set -u
LD=$1
D=~/fact_sft_amax; C=~/fact_sft_amax/code; LOG=$D/mug_pipeline_mix.txt
PY=/workspace/miniconda3/envs/fact/bin/python
TRAIN_G=4,5,6,7; STEPS=3194; TAG=mugM_nv; DR=$D/data/mugM
log() { echo "$(date '+%F %T') $*" >> $LOG; }
idle() { for g in "$@"; do
  [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .)" = "0" ] || return 1
  [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024 ] || return 1; done; }
wait_idle() { until idle "$@"; do sleep 120; done; }
guard() {  # same stop rule as the E/R pipeline
  while true; do
    if nvidia-smi -q 2>/dev/null | grep -q "GPU requires reset" \
       || grep -aqsE "ErrorDeviceLost|CUDA error: out of memory|CUDA is unavailable|No CUDA GPUs are available" \
            $D/runs/$TAG/train.log $D/vae_mugM.log 2>/dev/null; then
      log "GUARD: GPU error detected, stopping our training"
      pkill -u $USER -f "[l]aunch_from_config"; pkill -u $USER -f "[c]ompute_vae_latents"
      touch $D/GUARD_STOPPED_MIX; exit 0
    fi
    sleep 60
  done
}
[ -e $DR ] || [ -e $D/runs/$TAG ] && { log "data root or run dir exists; refusing to overwrite"; exit 2; }
guard & GP=$!
log "start: list $LD"
cd /workspace/FACT
export PATH=/workspace/miniconda3/envs/fact/bin:/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=/workspace/FACT:/workspace/fact_sft_amax
for s in E_expert:expert:hanging_mug_expert R_recovery:expert:hanging_mug_recovery R_prefix:prefix:hanging_mug_failprefix; do
  IFS=: read lst key name <<< "$s"
  log "convert $name"
  $PY $C/npz_to_lerobot.py --list $LD/$lst.jsonl --key $key --out $DR/Clean/$name > $D/convert_mugM_$name.log 2>&1 \
    || { log "convert FAILED ($name)"; kill $GP; exit 1; }
  log "$(tail -1 $D/convert_mugM_$name.log)"
done
wait_idle 4 5 6 7
for name in hanging_mug_expert hanging_mug_recovery hanging_mug_failprefix; do
  log "t5 $name"
  CUDA_VISIBLE_DEVICES=4 python -m scripts.compute_t5_embedding --root $DR/Clean/$name \
    --wan_path models/Wan2.2-TI2V-5B-Diffusers --device cuda > $D/t5_mugM_$name.log 2>&1 || { log "t5 FAILED"; kill $GP; exit 1; }
done
log "vae mugM"
FACT_PROJECT_DIR=$D/runs/_vae_tmp_mugM python -m scripts.compute_vae_latents --config_module fact_sft_configs.mug_sft_novalue \
  --robotwin_root $DR --devices cuda:4,cuda:5,cuda:6,cuda:7 > $D/vae_mugM.log 2>&1 || { log "vae FAILED"; kill $GP; exit 1; }
log "$(grep -a 'Done in' $D/vae_mugM.log | tail -1)"
log "verify masks"
ROBOTWIN_ROOT=$DR FACT_PROJECT_DIR=$D/runs/_vae_tmp_mugM python $C/verify_masks.py fact_sft_configs.mug_sft_novalue \
  > $D/verify_masks_mugM.log 2>&1 || { log "mask verification FAILED (see verify_masks_mugM.log)"; kill $GP; exit 1; }
log "$(tail -2 $D/verify_masks_mugM.log | tr '\n' ' ')"
MULT='{"*": 1.0, "Clean/hanging_mug_failprefix": 0.5}'
wait_idle 4 5 6 7
log "train $TAG: $STEPS steps on GPUs $TRAIN_G, multipliers $MULT"
FACT_GPU_IDS=$TRAIN_G FACT_CKPT_INTERVAL=1000 FACT_DS_OFFLOAD=0 FACT_CONFIG_MODULE=fact_sft_configs.mug_sft_novalue \
  bash $C/train_mug_sft_amax_nv.sh $TAG $DR $STEPS "$MULT" >> $LOG 2>&1
log "train $TAG finished: $(tail -1 $D/ledger.jsonl)"
TP=$(ls -d $D/runs/$TAG/models/checkpoint_*_step_$STEPS/transformer 2>/dev/null | head -1)
[ -n "$TP" ] && rm -rf $D/runs/$TAG/models/checkpoint_*/pytorch_model && log "$TAG final transformer $TP; DeepSpeed states removed" \
  || log "no final transformer for $TAG"
kill $GP
log "all done"
