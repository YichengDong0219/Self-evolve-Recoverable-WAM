#!/bin/bash
# hanging_mug E vs R comparison on Amax, end to end (Claude, 2026-10-29). Amax variant of mug_pipeline.sh.
# usage: mug_pipeline_amax.sh <list_dir>      (list_dir from mug_sft_lists.py: E_expert / R_recovery / R_prefix.jsonl)
# Order: convert E and R -> T5 -> VAE -> train E (GPUs 4-7) -> eval E on GPU 8 in the background while R trains
# (GPUs 4-7) -> eval R on GPUs 4-7 -> done. Both arms: config fact_sft_configs/mug_sft_mask.py, 3194 steps, ckpt
# every 1000 + final, DeepSpeed states removed after training. R: recovery (action loss from curobo's first motion on)
# + failure prefixes (FACT failure channel), prefix multiplier 0.5 -> 1/3 of samples. Eval: seeds 160000-160119,
# --skip_future_state_value (eval_task_amax.sh), same protocol as the base model.
# Safety (10-29 GPU incident): before each GPU stage the GPUs must have no compute process and < 1 GB used; a guard
# loop stops everything of ours if a log shows OOM / DeviceLost / CUDA unavailable or a GPU reports "requires reset".
set -u
LD=$1
D=~/fact_sft_amax; C=~/fact_sft_amax/code; LC=~/fact_local/code; LOG=$D/mug_pipeline.txt
PY=/workspace/miniconda3/envs/fact/bin/python
TRAIN_G=4,5,6,7; EVAL_E_G=8; STEPS=3194
log() { echo "$(date '+%F %T') $*" >> $LOG; }
idle() { for g in "$@"; do
  [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .)" = "0" ] || return 1
  [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024 ] || return 1; done; }
wait_idle() { until idle "$@"; do sleep 120; done; }
guard() {  # runs in the background for the whole pipeline
  while true; do
    if nvidia-smi -q 2>/dev/null | grep -q "GPU requires reset" \
       || grep -aqsE "ErrorDeviceLost|CUDA error: out of memory|CUDA is unavailable|No CUDA GPUs are available" \
            $D/runs/mug{E,R}/train.log ~/fact_local/eval_mug/mug{E,R}/*.log 2>/dev/null; then
      log "GUARD: GPU error detected, stopping our processes"
      pkill -u $USER -f "[l]aunch_from_config"; pkill -u $USER -f "[e]val_task_amax"
      pkill -u $USER -f "[i]nference_server_probe"; pkill -u $USER -f "[g]0_driver.py rollout"
      touch $D/GUARD_STOPPED; exit 0
    fi
    sleep 60
  done
}
[ -e $D/data/mugE ] || [ -e $D/data/mugR ] && { log "data roots exist; refusing to overwrite"; exit 2; }
guard & GP=$!
log "start: list $LD"
cd /workspace/FACT
export PATH=/workspace/miniconda3/envs/fact/bin:/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=/workspace/FACT:/workspace/fact_sft_amax
for s in E:E_expert:expert:hanging_mug_expert R:R_recovery:expert:hanging_mug_recovery R:R_prefix:prefix:hanging_mug_failprefix; do
  IFS=: read arm lst key name <<< "$s"
  log "convert $arm $name"
  $PY $C/npz_to_lerobot.py --list $LD/$lst.jsonl --key $key --out $D/data/mug$arm/Clean/$name > $D/convert_mug${arm}_$name.log 2>&1 \
    || { log "convert FAILED ($name)"; kill $GP; exit 1; }
  log "$(tail -1 $D/convert_mug${arm}_$name.log)"
done
wait_idle 4 5 6 7
for s in E:hanging_mug_expert R:hanging_mug_recovery R:hanging_mug_failprefix; do
  IFS=: read arm name <<< "$s"
  log "t5 $name"
  CUDA_VISIBLE_DEVICES=4 python -m scripts.compute_t5_embedding --root $D/data/mug$arm/Clean/$name \
    --wan_path models/Wan2.2-TI2V-5B-Diffusers --device cuda > $D/t5_$name.log 2>&1 || { log "t5 FAILED"; kill $GP; exit 1; }
done
for arm in E R; do
  log "vae mug$arm"
  FACT_PROJECT_DIR=$D/runs/_vae_tmp_mug$arm python -m scripts.compute_vae_latents --config_module fact_sft_configs.mug_sft_mask \
    --robotwin_root $D/data/mug$arm --devices cuda:4,cuda:5,cuda:6,cuda:7 > $D/vae_mug$arm.log 2>&1 || { log "vae FAILED"; kill $GP; exit 1; }
  log "$(grep -a 'Done in' $D/vae_mug$arm.log | tail -1)"
done
train() {  # $1 arm, $2 multipliers
  wait_idle 4 5 6 7
  log "train mug$1: $STEPS steps on GPUs $TRAIN_G, multipliers $2"
  FACT_GPU_IDS=$TRAIN_G FACT_CKPT_INTERVAL=1000 FACT_DS_OFFLOAD=0 bash $C/train_mug_sft_amax.sh mug$1 $D/data/mug$1 $STEPS "$2" >> $LOG 2>&1
  log "train mug$1 finished: $(tail -1 $D/ledger.jsonl)"
  TP=$(ls -d $D/runs/mug$1/models/checkpoint_*_step_$STEPS/transformer 2>/dev/null | head -1)
  [ -n "$TP" ] || { log "no final transformer for mug$1"; return 1; }
  rm -rf $D/runs/mug$1/models/checkpoint_*/pytorch_model && log "mug$1 final transformer $TP; DeepSpeed states removed"
}
evalm() {  # $1 arm, rest: GPUs (one seed shard each)
  local arm=$1; shift; local n=$#; local i=0
  local TP=$(ls -d $D/runs/mug$arm/models/checkpoint_*_step_$STEPS/transformer | head -1)
  for g in "$@"; do awk -v i=$i -v n=$n 'NR%n==i' ~/fact_local/eval_mug/seeds_all.txt > ~/fact_local/eval_mug/seeds_mug${arm}_$g.txt; i=$((i+1)); done
  wait_idle "$@"
  log "eval mug$arm on GPUs $* ($TP)"
  local pids=""
  for g in "$@"; do
    bash $LC/eval_task_amax.sh $g mug$arm $TP ~/fact_local/eval_mug/seeds_mug${arm}_$g.txt hanging_mug ~/fact_local/eval_mug \
      > ~/fact_local/eval_mug/eval_mug${arm}_gpu$g.log 2>&1 &
    pids="$pids $!"
  done
  wait $pids  # only these evals (the guard loop is also a child of this shell)
  log "eval mug$arm finished: $(ls ~/fact_local/eval_mug/mug$arm/*.json 2>/dev/null | wc -l) results"
}
train E '{"*": 1.0}' || { kill $GP; exit 1; }
evalm E $EVAL_E_G &
EP=$!
train R '{"*": 1.0, "Clean/hanging_mug_failprefix": 0.5}' || { kill $GP; exit 1; }
evalm R 4 5 6 7
wait $EP
log "all done"
kill $GP
