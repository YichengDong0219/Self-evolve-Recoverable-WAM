#!/bin/bash
# Self-sampling pipeline k of 2 (Claude, 2026-10-27). Cells from the frozen grid in grid order; index % 2 == k.
# Budget guard: GPU-hours = 4 cards x wall since T0 + 0.61 (IMG labels); no new cell once projected >= 15.5.
set -u
K=$1; G=$2; P=$3
B=~/fact_selfsample_R92; D=~/fact_g0_claude/code; T0=$(cat $B/T0_unix)
python3 -c "
import json
g=json.load(open('$B/SELF_SAMPLE_GRID_R93.json'))
for i,c in enumerate(g['cells']):
    if i%2==$K: print(c['name'],c['source'],c['branch_step'],c['stream_repeat'])
" > $B/pipeline_$K.todo
while read -r name src step rep; do
  used=$(python3 -c "import time;print(round(4*(time.time()-$T0)/3600+0.61,2))")
  if python3 -c "import sys;sys.exit(0 if $used>=15.5 else 1)"; then
    echo "$(date +%T) p$K BUDGET_STOP before $name (projected $used GPUh)" >> $B/SS_progress.txt; break
  fi
  bash $D/run_cell.sh $G $P $name $src $step $rep > /dev/null
  python3 -c "
import json,os
p='$B/cells/$name.json'
if os.path.exists(p):
    d=json.load(open(p)); print('$(date +%T) p$K', '$name', 'SUCCESS' if d['success'] else 'FAIL', d['steps'])
else: print('$(date +%T) p$K', '$name', 'NO_OUTPUT')" >> $B/SS_progress.txt
done < $B/pipeline_$K.todo
echo "$(date +%T) p$K done" >> $B/SS_progress.txt
