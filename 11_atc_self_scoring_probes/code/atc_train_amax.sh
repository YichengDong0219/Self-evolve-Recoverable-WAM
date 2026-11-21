#!/bin/bash
# ATC A1 pilot training (Claude, 2026-11-21; user approved "第2步批准"). Protocol: PROTOCOL.md section A1.
# R2 init; consequence-only fine-tune (future state 0.4 + value 0.4; action 0; visual 0) on the 1208 train-state chunks
# of the 11-14 chunk bank; value target raw = 1 - p per candidate (atc_patch + ATC_VALUE_TABLE); frame-0 windows only
# (root_only_patch); test-state episodes excluded from sampling (meta/atc_exclude_episodes.json).
# Recipe as R2/E100 SFT: LR 1e-5 / 1e-4, warmup 100 + cosine, ZeRO-2 + CPU offload; effective batch 16 (2 x 2 x 4).
# GPUs: two FULLY idle Amax 4090s (smi 4-8), re-checked right before launch; never the RTX PRO 6000.
# usage: setsid nohup bash atc_train_amax.sh smoke|full > /dev/null 2>&1 < /dev/null &   (log T/logs/a1_<mode>.txt)
set -u
MODE=${1:?smoke|full}
T=/workspace/fact_local/atc_probes_1121; D=/workspace/fact_sft_amax
DR=$T/wm_value/data; TASK=$DR/Clean/place_mouse_pad_chunks
R2=$D/ckpts_mouse_r100v2/mouseR100v2_nv_step1573
if [ "$MODE" = smoke ]; then STEPS=3; CI=1000; TAG=atcA1smoke; else STEPS=300; CI=100; TAG=atcA1; fi
RD=$D/runs/$TAG; K=$D/ckpts_atc_a1; LOG=$T/logs/a1_$MODE.txt
REF=$(stat -c %s /workspace/FACT/models/fact-wam/transformer/diffusion_pytorch_model.bin)
mkdir -p $T/logs
exec 9>$T/logs/a1_$MODE.lock; flock -n 9 || { echo "a1 $MODE lock held"; exit 1; }
log() { echo "$(date '+%F %T') $*" >> $LOG; }
idle() { [ -z "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader)" ] && \
         [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024 ]; }
[ -e $RD ] && { log "run dir $RD exists; refusing"; exit 2; }
for f in $TASK/meta/root_only.json $TASK/meta/atc_exclude_episodes.json $T/atc_value_table.json $R2/diffusion_pytorch_model.bin \
         $D/fact_sft_configs/atc_value.py $D/fact_sft_configs/atc_patch.py; do
  [ -e $f ] || { log "missing $f; refusing"; exit 2; }
done
G=""
for try in 1 2 3; do
  f=(); for g in 4 5 6 7 8; do case "$(nvidia-smi -i $g --query-gpu=name --format=csv,noheader)" in *4090*) idle $g && f+=($g);; esac; done
  if [ ${#f[@]} -ge 2 ]; then sleep 20; idle ${f[0]} && idle ${f[1]} && G=${f[0]},${f[1]} && break; fi
  sleep 60
done
[ -z "$G" ] && { log "no two idle 4090s; not started"; exit 6; }
log "start $MODE: $STEPS steps on smi $G, init $R2"
PY=/workspace/miniconda3/envs/fact/bin/python
cd /workspace/FACT
export PATH=/workspace/miniconda3/envs/fact/bin:/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4
export PYTHONPATH=/workspace/FACT:$D HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
guard() {
  while kill -0 $1 2>/dev/null; do
    if nvidia-smi -q 2>/dev/null | grep -q "GPU requires reset" \
       || grep -aqsE "ErrorDeviceLost|CUDA is unavailable|No CUDA GPUs are available|no kernel image|illegal memory access" $RD/train.log 2>/dev/null; then
      log "GUARD: GPU error, stopping our training (pgid $1)"; kill -TERM -- -$1; touch $T/logs/A1_GUARD_STOPPED_$MODE; return 0
    fi
    sleep 60
  done
}
snap() {
  mkdir -p $K
  while true; do
    for src in $RD/models/checkpoint_*_step_*/transformer; do
      [ -d "$src" ] || continue
      step=$(echo "$src" | sed -E 's/.*_step_([0-9]+)\/transformer/\1/'); dst=$K/${TAG}_step$step
      [ -e $dst ] && continue
      f=$src/diffusion_pytorch_model.bin; [ -e $f ] && [ -e $src/config.json ] || continue
      s1=$(stat -c %s $f 2>/dev/null); sleep 60; s2=$(stat -c %s $f 2>/dev/null)
      [ -n "$s1" ] && [ "$s1" = "$s2" ] && [ "$s1" -ge $((REF * 98 / 100)) ] || continue
      rm -rf $dst.tmp && cp -r $src $dst.tmp && cmp -s $f $dst.tmp/diffusion_pytorch_model.bin && mv $dst.tmp $dst && log "snap $dst" || log "snap FAILED $src"
    done
    [ -e $K/${TAG}_step$STEPS ] && exit 0
    sleep 30
  done
}
mkdir -p $RD; date +%s > $RD/start_unix
[ "$MODE" = full ] && { snap & SP=$!; }
WANDB_MODE=offline WANDB_DIR=$RD/wandb PYTHONUNBUFFERED=1 ROBOTWIN_ROOT=$DR FACT_PROJECT_DIR=$RD FACT_MAX_STEPS=$STEPS \
FACT_GPU_IDS=$G FACT_DATASET_MULTIPLIERS='{"*": 1.0}' FACT_CKPT_INTERVAL=$CI FACT_DS_OFFLOAD=1 FACT_BATCH_PER_GPU=2 FACT_GRAD_ACCUM=4 \
FACT_INIT_CHECKPOINT=$R2 ATC_VALUE_TABLE=$T/atc_value_table.json \
  setsid $PY -c "from fact_train import launch_from_config; launch_from_config('fact_sft_configs.atc_value.config')" > $RD/train.log 2>&1 &
TP=$!; echo $TP > $RD/train.pgid
guard $TP & GP=$!
wait $TP; rc=$?
log "train rc=$rc wall $(( $(date +%s) - $(cat $RD/start_unix) )) s"
if [ "$MODE" = full ]; then
  for i in $(seq 30); do [ -e $K/${TAG}_step$STEPS ] && break; [ $rc -ne 0 ] && break; sleep 60; done
  kill $SP 2>/dev/null
  [ -e $K/${TAG}_step$STEPS ] && rm -rf $RD/models/checkpoint_*/pytorch_model && log "final transformer copied; DeepSpeed states removed"
fi
kill $GP 2>/dev/null
echo $rc > $T/logs/a1_$MODE.rc
log "end"
