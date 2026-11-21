#!/bin/bash
# Launch one TTS round on the idle Amax 4090s (Claude, 2026-11-21): one worker per GPU, seeds split 5 ways.
# usage: launch_round_amax.sh <r0|r1> <random|critic> [critic host:port]
set -u
RND=$1; SEL=$2; CA=${3:-127.0.0.1:8650}; M=/workspace/fact_local/mug_loop_1121; O=$M/$RND
mkdir -p $O
w=0
for g in 4 5 6 7 8; do
  if [ -z "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader)" ] && \
     [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024 ]; then
    setsid nohup bash $M/code/run_worker_amax.sh $g $M/seeds/${RND}_w$w.txt $O $SEL $CA > $O/worker_gpu$g.out 2>&1 < /dev/null &
    echo "worker $w on smi $g"
  else
    echo "smi $g busy; worker $w NOT started"
  fi
  w=$((w + 1))
done
