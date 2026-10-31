#!/bin/bash
# Restart the local side of the v7 stream with a new local worker list (Claude, 2026-10-29).
# usage: local_restart.sh <local_workers e.g. 3,3,3>
# Kept in a file so the kill patterns never match the invoking shell's own command line.
set -u
LW=$1
for p in $(pgrep -u $USER -f "[m]ug_stream_multi.sh"); do kill $p; done
for p in $(pgrep -u $USER -f "[s]im_select.py"); do kill $p; done
sleep 2
for p in $(pgrep -u $USER -f "[t]akeover_bench.py"); do kill $p; done
sleep 5
echo "left: $(pgrep -u $USER -f '[t]akeover_bench.py|[s]im_select|[m]ug_stream' | wc -l)"
cd /workspace/fact_local
AMAX_W=4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6  # GPUs 7,8 lost CUDA 10-29 07:07
setsid nohup bash code/mug_stream_multi.sh "$LW" $AMAX_W "$(seq -s, 0 12)/31" "$(seq -s, 13 30)/31" 8 2 \
  > bench/stream_v7_runner_$(date +%H%M).log 2>&1 < /dev/null &
echo "restarted local workers $LW"
