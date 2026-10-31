#!/bin/bash
# Focus review: seeds where expert-only step 3000 FAILED but Mix step 3000 SUCCEEDED (Claude, 2026-10-31, user request).
# Waits for both renders of each seed on the cluster, pulls them, and stacks Mix (top) over E (bottom); the shorter
# video holds its last frame. usage: make_focus_pairs.sh <out dir> <seed,...>
set -u
D=$1; SEEDS=${2//,/ }; C=user@CLUSTER_HOST; Q=/workspace_cluster/fact_sft/videos_eval
mkdir -p $D/single
todo=$SEEDS
while [ -n "$todo" ]; do
  left=""
  for s in $todo; do
    m=$Q/mugM_nv_step3000/hanging_mug_seed${s}_SUCCESS.mp4; e=$Q/mugE_nv_step3000/hanging_mug_seed${s}_FAIL.mp4
    if ssh -o BatchMode=yes $C "test -e $m -a -e $e"; then
      scp -q $C:$m $D/single/M3000_seed${s}_SUCCESS.mp4; scp -q $C:$e $D/single/E3000_seed${s}_FAIL.mp4
      ffmpeg -loglevel error -y -i $D/single/M3000_seed${s}_SUCCESS.mp4 -i $D/single/E3000_seed${s}_FAIL.mp4 -filter_complex \
        "[0:v]drawtext=text='Mix step3000 (SUCCESS)':x=w-260:y=8:fontsize=16:fontcolor=yellow:box=1:boxcolor=black@0.6,tpad=stop=-1:stop_mode=clone[a];[1:v]drawtext=text='Expert-only step3000 (FAIL)':x=w-290:y=8:fontsize=16:fontcolor=yellow:box=1:boxcolor=black@0.6[b];[a][b]vstack=shortest=1[v]" \
        -map "[v]" -vcodec libx264 -crf 23 -pix_fmt yuv420p -shortest $D/seed${s}_Mix_vs_E.mp4 \
        || ffmpeg -loglevel error -y -i $D/single/M3000_seed${s}_SUCCESS.mp4 -i $D/single/E3000_seed${s}_FAIL.mp4 -filter_complex \
        "[0:v]tpad=stop=-1:stop_mode=clone[a];[a][1:v]vstack=shortest=1[v]" -map "[v]" -vcodec libx264 -crf 23 -pix_fmt yuv420p -shortest $D/seed${s}_Mix_vs_E.mp4
      echo "$(date '+%T') seed $s done"
    else left="$left $s"; fi
  done
  todo=$(echo $left)
  [ -n "$todo" ] && sleep 120
done
echo "all done"
