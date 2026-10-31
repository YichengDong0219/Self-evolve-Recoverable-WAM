#!/bin/bash
# hanging_mug FACT failure-state collection, one self-contained pipeline per GPU (Claude, 2026-10-28).
# usage: collect_mug_failures.sh <gpu> <start_seed> [target_failures=300]
# The GPU hosts its own FACT server (deployment config: prefix cache on, action-only, N=1) and the simulator.
# Every rollout (success or failure) is kept; takeover/recovery runs elsewhere (local sim_select.py). All pipelines
# stop once ~/fact_collect_mug/failures.txt holds target_failures lines. Seeds: 140000+ (110000-117999 used by the
# turn_switch collection, 130000-130119 by the SFT evaluation).
set -u
G=$1; S=$2; TARGET=${3:-300}; P=$((8130 + G)); MAX_SEEDS=600
O=~/fact_collect_mug; C=~/fact_g0_claude/code; R=~/fact_g0_claude/runs
mkdir -p $O/rollouts; touch $O/failures.txt
L=$O/progress_gpu$G.txt
bash $C/start_server.sh MUGCOL_gpu$G $G $P --enable_prefix_cache
for i in $(seq 90); do grep -q "ready and listening" $R/server_MUGCOL_gpu$G/server.log && break; sleep 10; done
grep -q "ready and listening" $R/server_MUGCOL_gpu$G/server.log || { echo "$(date +%T) server not ready" >> $L; kill $(cat $R/server_MUGCOL_gpu$G/pid); exit 1; }
echo "$(date +%T) server ready on port $P" >> $L
cd /workspace_cluster/RoboTwin-FACT
export PATH=/workspace_cluster/miniconda3/envs/robotwin/bin:/workspace_cluster/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_EXTENSIONS_DIR=/workspace_cluster/.cache/openwam-robotwin-torch-extensions
export CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 FACT_PROBE_DIR=$C PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/RoboTwin-FACT:$C
PY=/workspace_cluster/miniconda3/envs/robotwin/bin/python
for n in $(seq $MAX_SEEDS); do
  [ $(wc -l < $O/failures.txt) -ge $TARGET ] && { echo "$(date +%T) target reached" >> $L; break; }
  name=MUG_hanging_mug_from$S
  t0=$(date +%s)
  timeout 2400 $PY $C/g0_driver.py rollout --task hanging_mug --config demo_clean --start-seed $S --max-tries 30 \
    --port $P --out $O/rollouts/$name.json > $O/rollouts/$name.log 2>&1
  rc=$?; echo "{\"name\":\"$name\",\"gpu\":\"$G\",\"rc\":$rc,\"wall_s\":$(( $(date +%s)-t0 ))}" >> $O/ledger.jsonl
  [ -e $O/rollouts/$name.json ] || { echo "$(date +%T) $name no output (rc $rc); skipping 1 seed" >> $L; S=$((S+1)); continue; }
  read seed succ < <($PY -c "import json;d=json.load(open('$O/rollouts/$name.json'));print(d['env_seed'],d['success'])")
  S=$((seed + 1))
  echo "$(date +%T) seed $seed success=$succ" >> $L
  [ "$succ" = "False" ] && echo "$seed $name gpu$G" >> $O/failures.txt
done
p=$(cat $R/server_MUGCOL_gpu$G/pid); kill $p; sleep 5; ps -p $p > /dev/null && kill -9 $p
echo "$(date +%T) pipeline done, server stopped" >> $L
