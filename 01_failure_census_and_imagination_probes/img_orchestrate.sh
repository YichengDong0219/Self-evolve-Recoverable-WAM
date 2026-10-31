#!/bin/bash
set -u
R=~/fact_g0_claude/runs; D=~/fact_g0_claude/code
stop(){ p=$(cat $R/server_$1/pid); kill $p; sleep 5; ps -p $p >/dev/null && kill -9 $p; date +%s > $R/server_$1/stop_unix; echo "$(date +%T) server $1 stopped" >> $R/IMG_progress.txt; }
# pipeline A: new return_images server on GPU0:8093, sim GPU1
( bash $D/start_server.sh IMG_A 0 8093 --return_images
  for i in $(seq 90); do grep -q "ready and listening" $R/server_IMG_A/server.log && break; sleep 10; done
  bash $D/img_census.sh 1 8093 demo_clean 20 hanging_mug
  stop IMG_A ) &
# pipeline B: existing return_images server C_imagine on GPU2:8094, sim GPU3
( bash $D/img_census.sh 3 8094 demo_clean 20 turn_switch click_bell
  stop C_imagine ) &
wait
echo "$(date +%T) ALL IMG DONE" >> $R/IMG_progress.txt
