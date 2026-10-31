#!/bin/bash
# After an SFT run: keep transformer snapshots, drop DeepSpeed states, then evaluate on held-out seeds (Claude, 2026-10-28).
# usage: post_train_eval.sh <arm_tag> <max_steps> <mid_step> [also_base=1]
# - copies models/checkpoint_*_step_<mid_step>/transformer as soon as it is complete (it is rotated away later)
# - when training ends: copies the final transformer, deletes the ~68 GB DeepSpeed state of the run
# - eval on ~/fact_sft/eval/seeds_half{0,1}.txt: arm on GPUs 2,3; released FACT ("base") on GPUs 0,1 if also_base=1
set -u
TAG=$1; STEPS=$2; MID=$3; BASE=${4:-1}
D=~/fact_sft/runs/$TAG; K=~/fact_sft/ckpts; C=~/fact_g0_claude/code; L=~/fact_sft/post_$TAG.txt
mkdir -p $K
snap() {  # $1 step -> copy transformer once its .bin size is stable
  local src
  src=$(ls -d $D/models/checkpoint_*_step_$1/transformer 2>/dev/null | head -1)
  [ -n "$src" ] && [ -e $src/diffusion_pytorch_model.bin ] || return 1
  local s1 s2; s1=$(stat -c %s $src/diffusion_pytorch_model.bin); sleep 30; s2=$(stat -c %s $src/diffusion_pytorch_model.bin)
  [ "$s1" = "$s2" ] || return 1
  cp -r $src $K/${TAG}_step$1 && echo "$(date +%T) saved $K/${TAG}_step$1" >> $L
}
until [ -e $K/${TAG}_step$MID ] || [ -e $D/stop_unix ]; do snap $MID || sleep 60; done
until [ -e $D/stop_unix ]; do sleep 60; done
rc=$(tail -1 ~/fact_sft/ledger.jsonl)
echo "$(date +%T) training ended: $rc" >> $L
[ -e $K/${TAG}_step$STEPS ] || snap $STEPS || { echo "$(date +%T) no final transformer; stop" >> $L; exit 1; }
rm -rf $D/models/checkpoint_*/pytorch_model && echo "$(date +%T) removed DeepSpeed states" >> $L
idle() {  # GPU $1 has no compute process (ours or anyone else's) and < 1 GB in use
  local apps used
  apps=$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | wc -l)
  used=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$apps" = "0" ] && [ "$used" -lt 1024 ]
}
for g in 0 1 2 3; do
  until idle $g; do echo "$(date +%T) GPU $g busy, waiting" >> $L; sleep 300; done
done
for g in 2 3; do
  bash $C/eval_switch.sh $g $TAG $K/${TAG}_step$STEPS ~/fact_sft/eval/seeds_half$((g-2)).txt > ~/fact_sft/eval_${TAG}_gpu$g.log 2>&1 &
done
if [ "$BASE" = "1" ]; then
  for g in 0 1; do
    bash $C/eval_switch.sh $g base /workspace_cluster/FACT/models/fact-wam/transformer ~/fact_sft/eval/seeds_half$g.txt > ~/fact_sft/eval_base_gpu$g.log 2>&1 &
  done
fi
wait
echo "$(date +%T) eval done" >> $L
