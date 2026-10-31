#!/bin/bash
# Top-up loop for the hanging_mug paired set (Claude, 2026-10-29). Every 10 min: pull the Amax selections, run expert
# demos for newly selected states on local GPUs, count usable pairs (mug_sft_lists.py rules); stop the Amax
# sim_select once N pairs are available. usage: topup_until_paired.sh <local_gpus e.g. 4,5,6,7> [N=100]
set -u
G=$1; N=${2:-100}
L=/workspace/fact_local; A=user@AMAX_HOST; PY=/workspace/miniconda3/envs/RoboTwin/bin/python
S=/tmp/scratch
cd $L
while true; do
  rsync -a $A:fact_local/bench/SEL_v7/ bench/SEL_v7/
  $PY -u code/expert_batch.py bench/SEL_v7 expert_mug $G >> expert_mug_topup.log 2>&1
  rm -rf $S/lists_count
  n=$($PY code/mug_sft_lists.py bench/SEL_v7 expert_mug $S/lists_count 1000 | grep "paired available" | awk '{print $3}')
  echo "$(date +%T) paired $n"
  if [ "${n:-0}" -ge $N ]; then
    ssh $A 'pkill -u $USER -f "[s]im_select.py"; sleep 3; pkill -u $USER -f "[t]imeout 2400 python"; sleep 2; pkill -u $USER -f "[t]akeover_bench.py"'
    rsync -a $A:fact_local/bench/SEL_v7/ bench/SEL_v7/
    echo "PAIRED $n >= $N, Amax stopped"; exit 0
  fi
  sleep 600
done
