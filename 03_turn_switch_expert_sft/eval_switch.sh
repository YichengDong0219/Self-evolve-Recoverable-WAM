#!/bin/bash
# turn_switch evaluation of a FACT transformer on an explicit held-out seed list (Claude, 2026-10-28).
# usage: eval_switch.sh <gpu> <tag> <transformer_path> <seed_file>
# One GPU hosts the FACT server (deployment config: prefix cache on, action-only, N=1) and the simulator.
# Each seed is run exactly once with --max-tries 1 (same seed retried up to 3x only if the expert check is flaky),
# so every model is scored on the same scenes; seeds whose expert check never passes are logged as INVALID.
set -u
G=$1; TAG=$2; TP=$3; SEEDS=$4; P=$((8120 + G))
O=~/fact_sft/eval/$TAG; C=~/fact_g0_claude/code; R=~/fact_g0_claude/runs
mkdir -p $O; L=$O/progress_gpu$G.txt
bash $C/start_server.sh EVAL_${TAG}_gpu$G $G $P --enable_prefix_cache --transformer_path $TP
for i in $(seq 90); do grep -q "ready and listening" $R/server_EVAL_${TAG}_gpu$G/server.log && break; sleep 10; done
grep -q "ready and listening" $R/server_EVAL_${TAG}_gpu$G/server.log || { echo "$(date +%T) server not ready" >> $L; kill $(cat $R/server_EVAL_${TAG}_gpu$G/pid); exit 1; }
echo "$(date +%T) server ready ($TP)" >> $L
cd /workspace_cluster/RoboTwin-FACT
export PATH=/workspace_cluster/miniconda3/envs/robotwin/bin:/workspace_cluster/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_EXTENSIONS_DIR=/workspace_cluster/.cache/openwam-robotwin-torch-extensions
export CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 FACT_PROBE_DIR=$C PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/RoboTwin-FACT:$C
PY=/workspace_cluster/miniconda3/envs/robotwin/bin/python
for s in $(cat $SEEDS); do
  out=$O/turn_switch_seed$s
  [ -e $out.json ] && continue
  for try in 1 2 3; do
    t0=$(date +%s)
    timeout 1500 $PY $C/g0_driver.py rollout --task turn_switch --config demo_clean --start-seed $s --max-tries 1 \
      --port $P --out $out.json > $out.try$try.log 2>&1
    echo "{\"name\":\"eval_${TAG}_$s\",\"gpu\":\"$G\",\"rc\":$?,\"wall_s\":$(( $(date +%s)-t0 ))}" >> $O/ledger.jsonl
    [ -e $out.json ] && break
  done
  r=$($PY -c "import json,os;p='$out.json';print(json.load(open(p))['success'] if os.path.exists(p) else 'INVALID')")
  echo "$(date +%T) seed $s $r" >> $L
done
p=$(cat $R/server_EVAL_${TAG}_gpu$G/pid); kill $p; sleep 5; ps -p $p > /dev/null && kill -9 $p
echo "$(date +%T) done, server stopped" >> $L
