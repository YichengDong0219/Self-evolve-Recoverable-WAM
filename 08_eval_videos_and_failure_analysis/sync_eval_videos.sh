#!/bin/bash
# Pull rendered eval videos from the cluster every 5 min until all queued items are in the ledger, then write the index
# (Claude, 2026-10-31). usage: sync_eval_videos.sh <local dir>
set -u
D=$1; C=user@CLUSTER_HOST; Q=/workspace_cluster/fact_sft/videos_eval
mkdir -p $D
while true; do
  n=$(ssh -o BatchMode=yes $C "cat $Q/ledger.jsonl | wc -l; wc -l < $Q/queue.txt" | tr '\n' ' ')
  rsync -rt --include='*/' --exclude='*.tmp.mp4' --include='*.mp4' --include='ledger.jsonl' --include='queue.txt' --exclude='*' $C:$Q/ $D/ 2>/dev/null
  set -- $n; [ "$1" -ge "$2" ] && break
  sleep 300
done
rsync -rt --include='*/' --exclude='*.tmp.mp4' --include='*.mp4' --include='*.log' --include='ledger.jsonl' --exclude='.claim' --exclude='*' $C:$Q/ $D/ 2>/dev/null
/workspace/miniconda3/envs/RoboTwin/bin/python /workspace/fact_local/code/eval_video_index.py $D
