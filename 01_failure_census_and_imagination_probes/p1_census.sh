#!/bin/bash
# P1 failure census (Claude, 2026-10-27): per task config, N valid episodes from consecutive expert-valid seeds.
# usage: p1_census.sh <gpu> <task> <config> <n_episodes> [start_seed]
set -u
G=$1; T=$2; C=$3; N=$4; S=${5:-100000}
D=~/fact_g0_claude/code; R=~/fact_g0_claude/runs
for i in $(seq 0 $((N-1))); do
  name=P1_${T}_${C}_$(printf %02d $i)
  [ -e $R/$name.json ] && { S=$(python3 -c "import json;print(json.load(open('$R/$name.json'))['env_seed']+1)"); continue; }
  bash $D/run_driver.sh $G $name rollout --task $T --config $C --start-seed $S --max-tries 30 > /dev/null
  [ -e $R/$name.json ] || { echo "$(date +%T) $name FAILED to produce output; stopping $T"; exit 1; }
  S=$(python3 -c "import json;d=json.load(open('$R/$name.json'));print(d['env_seed']+1);import sys;print('$name',d['env_seed'],d['success'],d['steps'],file=sys.stderr)" 2>> $R/P1_progress.txt)
done
echo "$(date +%T) done $T $C" >> $R/P1_progress.txt
