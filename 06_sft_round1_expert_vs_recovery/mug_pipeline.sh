#!/bin/bash
# hanging_mug E vs R comparison, one arm end to end on the 4x4090 (48 GB) cluster (Claude, 2026-10-29).
# usage: mug_pipeline.sh <E|R> <list_dir>
# list_dir (from mug_sft_lists.py, npz paths rewritten to ~/fact_sft/mug_npz): E_expert / R_recovery / R_prefix.jsonl
#   E  root data/mugE: Clean/hanging_mug_expert (expert demos)
#   R  root data/mugR: Clean/hanging_mug_recovery (recovery segments; action loss from curobo's first motion on,
#                        the settle frames before it train the world model only: action_mask_patch.py)
#                      + Clean/hanging_mug_failprefix (FACT failure prefixes, failure channel: no action loss, value
#                        penalty from the fall onset); sampling multiplier 0.5 on 100 + 100 episodes -> prefix 1/3
# Steps: 3194 for BOTH arms (user 10-29: equal steps = equal compute, LR schedule, world-model samples and action-head
# updates; FACT averages the action loss over supervised samples only). 3194 = 3 epochs of the E arm (34059 frames / 32).
# Same config (fact_sft_configs/mug_sft_mask.py = mug_sft.py + action_mask_patch, a no-op for E), LR, warmup and init for both arms. Waits for all 4 GPUs to be idle
# (no compute process of anyone) before each GPU stage. Then evaluates the final transformer on the same 120 seeds
# (160000-160119) as the base model, 4 GPUs.
set -u
ARM=$1; LD=$2
C=~/fact_g0_claude/code; D=~/fact_sft; ROOT=$D/data/mug$ARM; LOG=$D/mug_pipeline_$ARM.txt
PY=/workspace_cluster/miniconda3/envs/fact/bin/python
log() { echo "$(date +%T) $*" >> $LOG; }
idle_all() { for g in 0 1 2 3; do
  [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ] || return 1
  [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024 ] || return 1; done; }
wait_idle() { until idle_all; do sleep 120; done; }
[ -e $ROOT ] && { log "root exists: $ROOT"; exit 2; }
if [ $ARM = E ]; then SETS="E_expert:expert:hanging_mug_expert"; MULT='{"*": 1.0}'
else SETS="R_recovery:expert:hanging_mug_recovery R_prefix:prefix:hanging_mug_failprefix"
     MULT='{"*": 1.0, "Clean/hanging_mug_failprefix": 0.5}'; fi
for s in $SETS; do
  IFS=: read lst key name <<< "$s"
  log "convert $lst"
  $PY $C/npz_to_lerobot.py --list $LD/$lst.jsonl --key $key --out $ROOT/Clean/$name > $D/convert_mug${ARM}_$name.log 2>&1 \
    || { log "convert FAILED"; exit 1; }
  log "$(tail -1 $D/convert_mug${ARM}_$name.log)"
done
FR=$($PY -c "import json;print(json.load(open('$ROOT/Clean/$( [ $ARM = E ] && echo hanging_mug_expert || echo hanging_mug_recovery )/meta/info.json'))['total_frames'])")
STEPS=3194
log "frames in the action-supervised set: $FR; steps $STEPS (fixed, same for both arms)"
cd ~/FACT
export PATH=/workspace_cluster/miniconda3/envs/fact/bin:/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/fact_sft
wait_idle
for s in $SETS; do
  IFS=: read lst key name <<< "$s"
  log "t5 $name"
  CUDA_VISIBLE_DEVICES=0 python -m scripts.compute_t5_embedding --root $ROOT/Clean/$name --wan_path models/Wan2.2-TI2V-5B-Diffusers \
    --device cuda > $D/t5_mug${ARM}_$name.log 2>&1 || { log "t5 FAILED"; exit 1; }
done
log "vae"
FACT_PROJECT_DIR=$D/runs/_vae_tmp_mug$ARM python -m scripts.compute_vae_latents --config_module fact_sft_configs.mug_sft_mask \
  --robotwin_root $ROOT --devices cuda:0,cuda:1,cuda:2,cuda:3 > $D/vae_mug$ARM.log 2>&1 || { log "vae FAILED"; exit 1; }
log "$(grep -a 'Done in' $D/vae_mug$ARM.log | tail -1)"
wait_idle
log "train mug$ARM $STEPS steps, multipliers $MULT"
FACT_CKPT_INTERVAL=1000 FACT_DS_OFFLOAD=0 bash $C/train_mug_sft.sh mug$ARM $ROOT $STEPS "$MULT" >> $LOG 2>&1
log "train finished: $(tail -1 $D/ledger.jsonl)"
TP=$(ls -d $D/runs/mug$ARM/models/checkpoint_*/transformer 2>/dev/null | sort -V | tail -1)
[ -n "$TP" ] || { log "no final transformer found"; exit 1; }
log "final transformer $TP"
rm -rf $D/runs/mug$ARM/models/checkpoint_*/pytorch_model && log "removed DeepSpeed states (disk: $(df -h ~ | tail -1 | awk '{print $4}') free)"
wait_idle
for g in 0 1 2 3; do
  bash $C/eval_task.sh $g mug$ARM $TP $D/eval_mug/seeds_q$g.txt hanging_mug $D/eval_mug > $D/eval_mug/eval_mug${ARM}_gpu$g.log 2>&1 &
done
wait
log "eval finished"
