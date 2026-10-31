#!/bin/bash
# Imagination-recording census (Claude, 2026-10-27): N consecutive expert-valid episodes per task, server with
# --return_images (prefix cache off), then a side-by-side real|imagined video per episode.
# usage: img_census.sh <sim_gpu> <port> <config> <n> <task>...
set -u
G=$1; P=$2; C=$3; N=$4; shift 4
D=~/fact_g0_claude/code; R=~/fact_g0_claude/runs; L=$R/IMG_progress.txt
for T in "$@"; do
  S=100000
  for i in $(seq 0 $((N-1))); do
    name=IMG_${T}_${C}_$(printf %02d $i)
    if [ ! -e $R/$name.json ]; then
      bash $D/run_driver.sh $G $name rollout --task $T --config $C --start-seed $S --max-tries 30 --port $P > /dev/null
    fi
    [ -e $R/$name.json ] || { echo "$(date +%T) $name FAILED, stopping $T" >> $L; break; }
    S=$(python3 -c "import json;d=json.load(open('$R/$name.json'));print(d['env_seed']+1)")
    python3 -c "import json;d=json.load(open('$R/$name.json'));print('$(date +%T)','$name',d['env_seed'],'S' if d['success'] else 'F',d['steps'],'imag',d.get('imagined_decisions'))" >> $L
    bash $D/run_sbs.sh $G $name > /dev/null
  done
  echo "$(date +%T) done $T" >> $L
done
