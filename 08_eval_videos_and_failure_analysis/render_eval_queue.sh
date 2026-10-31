#!/bin/bash
# Render eval rollouts to mp4 by bit-exact replay with p1_render.py (Claude, 2026-10-31, user request).
# usage: render_eval_queue.sh <gpu> <worker id> [queue file, default $Q/queue.txt]   (run several; items are claimed with mkdir, so workers share one queue)
# queue = $Q/queue.txt lines "<tag> <rollout json>"; output $Q/<tag>/hanging_mug_seed<N>_<SUCCESS|FAIL>.mp4
set -u
G=$1; W=$2; Q=/workspace_cluster/fact_sft/videos_eval; QF=${3:-$Q/queue.txt}
cd /workspace_cluster/RoboTwin-FACT
export PATH=/workspace_cluster/miniconda3/envs/robotwin/bin:/workspace_cluster/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_EXTENSIONS_DIR=/workspace_cluster/.cache/openwam-robotwin-torch-extensions
export CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 FACT_PROBE_DIR=/workspace_cluster/fact_g0_claude/code
export PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/RoboTwin-FACT:/workspace_cluster/fact_g0_claude/code
while read tag js; do
  n=$(basename $js .json)
  mkdir $Q/.claim/${tag}__$n 2>/dev/null || continue
  ok=$(grep -q '"success": true' $js && echo SUCCESS || echo FAIL)
  out=$Q/$tag/${n}_$ok.mp4; mkdir -p $Q/$tag
  t0=$(date +%s)
  timeout 1500 python /workspace_cluster/fact_g0_claude/code/p1_render.py --rollout $js --out $out > $Q/$tag/$n.log 2>&1
  echo "{\"tag\":\"$tag\",\"rollout\":\"$n\",\"gpu\":$G,\"worker\":$W,\"rc\":$?,\"wall_s\":$(( $(date +%s)-t0 ))}" >> $Q/ledger.jsonl
done < $QF
