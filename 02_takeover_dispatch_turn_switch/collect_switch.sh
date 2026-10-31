#!/bin/bash
# turn_switch failure -> recovery collection, one self-contained pipeline per GPU (Claude, 2026-10-27).
# usage: collect_switch.sh <gpu> <start_seed> [target_total=200]
# The GPU hosts its own FACT server (deployment config: prefix cache on, action-only, N=1) and the simulator.
# Loop: FACT rollout on the next expert-valid seed -> if it fails, takeover with plans A, W1, W2 (stop at first
# SUCCESS). All pipelines stop once ~/fact_collect_switch/success.txt holds target_total lines.
set -u
G=$1; S=$2; TARGET=${3:-200}; P=$((8110 + G)); MAX_SEEDS=400
O=~/fact_collect_switch; C=~/fact_g0_claude/code; R=~/fact_g0_claude/runs
mkdir -p $O/rollouts $O/takeover; touch $O/success.txt
L=$O/progress_gpu$G.txt
bash $C/start_server.sh COL_gpu$G $G $P --enable_prefix_cache
for i in $(seq 90); do grep -q "ready and listening" $R/server_COL_gpu$G/server.log && break; sleep 10; done
grep -q "ready and listening" $R/server_COL_gpu$G/server.log || { echo "$(date +%T) server not ready" >> $L; kill $(cat $R/server_COL_gpu$G/pid); exit 1; }
echo "$(date +%T) server ready on port $P" >> $L
cd /workspace_cluster/RoboTwin-FACT
export PATH=/workspace_cluster/miniconda3/envs/robotwin/bin:/workspace_cluster/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_EXTENSIONS_DIR=/workspace_cluster/.cache/openwam-robotwin-torch-extensions
export CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 FACT_PROBE_DIR=$C PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/RoboTwin-FACT:$C
PY=/workspace_cluster/miniconda3/envs/robotwin/bin/python
for n in $(seq $MAX_SEEDS); do
  [ $(wc -l < $O/success.txt) -ge $TARGET ] && { echo "$(date +%T) target reached" >> $L; break; }
  name=COL_turn_switch_from$S
  t0=$(date +%s)
  timeout 1500 $PY $C/g0_driver.py rollout --task turn_switch --config demo_clean --start-seed $S --max-tries 30 \
    --port $P --out $O/rollouts/$name.json > $O/rollouts/$name.log 2>&1
  rc=$?; echo "{\"name\":\"$name\",\"gpu\":\"$G\",\"rc\":$rc,\"wall_s\":$(( $(date +%s)-t0 ))}" >> $O/ledger.jsonl
  [ -e $O/rollouts/$name.json ] || { echo "$(date +%T) $name rollout produced no output (rc $rc); skipping 1 seed" >> $L; S=$((S+1)); continue; }
  read seed succ < <($PY -c "import json;d=json.load(open('$O/rollouts/$name.json'));print(d['env_seed'],d['success'])")
  S=$((seed + 1))
  if [ "$succ" = "True" ]; then echo "$(date +%T) seed $seed policy SUCCESS" >> $L; continue; fi
  res=FAIL
  for p in A W1 W2; do
    out=$O/takeover/turn_switch_seed${seed}_$p
    t0=$(date +%s)
    timeout 900 $PY $C/collect_recovery.py --rollout $O/rollouts/$name.json --plan $p --out $out > $out.log 2>&1
    rc=$?; echo "{\"name\":\"takeover_${seed}_$p\",\"gpu\":\"$G\",\"rc\":$rc,\"wall_s\":$(( $(date +%s)-t0 ))}" >> $O/ledger.jsonl
    st=$($PY -c "import json,os;p='$out.json';print(json.load(open(p))['status'] if os.path.exists(p) else 'NO_OUTPUT')")
    echo "$(date +%T) seed $seed policy FAIL -> takeover $p $st" >> $L
    [ "$st" = "SUCCESS" ] && { res=SUCCESS; echo "$seed $p gpu$G" >> $O/success.txt; break; }
  done
done
p=$(cat $R/server_COL_gpu$G/pid); kill $p; sleep 5; ps -p $p > /dev/null && kill -9 $p
date +%s > $R/server_COL_gpu$G/stop_unix
echo "$(date +%T) pipeline done, server stopped" >> $L
