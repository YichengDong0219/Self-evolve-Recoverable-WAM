#!/bin/bash
# After a GPU's first-pass worker finishes, rerun the same seed list once (retries seeds without json/invalid). Claude 11-21.
set -u
G=$1; RND=$2; SEL=$3; W=$4; M=/workspace/fact_local/mug_loop_1121; O=$M/$RND
until grep -q "done, server stopped" $O/progress_gpu$G.txt 2>/dev/null; do sleep 60; done
sleep 30
bash $M/code/run_worker_amax.sh $G $M/seeds/${RND}_w$W.txt $O $SEL >> $O/worker_gpu$G.second.out 2>&1
echo "$(date +%T) second pass done" >> $O/progress_gpu$G.txt
