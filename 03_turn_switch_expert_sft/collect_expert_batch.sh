#!/bin/bash
# Expert demos on the recovery-set scenes, sharded over GPUs (Claude, 2026-10-28). No model server.
# usage: collect_expert_batch.sh <gpu> <shard> <n_shards>
set -u
G=$1; K=$2; N=$3
O=~/fact_collect_switch_expert; C=~/fact_g0_claude/code; M=~/fact_collect_switch/MANIFEST_turn_switch_recovery_v1.jsonl
mkdir -p $O/demos; L=$O/progress_gpu$G.txt
cd /workspace_cluster/RoboTwin-FACT
export PATH=/workspace_cluster/miniconda3/envs/robotwin/bin:/workspace_cluster/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_EXTENSIONS_DIR=/workspace_cluster/.cache/openwam-robotwin-torch-extensions
export CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 FACT_PROBE_DIR=$C PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/RoboTwin-FACT:$C
PY=/workspace_cluster/miniconda3/envs/robotwin/bin/python
$PY -c "import json;[print(r['env_seed'],r['rollout']) for i,r in enumerate(map(json.loads,open('$M'))) if i%$N==$K]" |
while read -r seed roll; do
  out=$O/demos/turn_switch_seed${seed}_expert
  [ -e $out.json ] && continue
  t0=$(date +%s)
  timeout 900 $PY $C/collect_expert_demo.py --rollout ~/fact_collect_switch/$roll --out $out > $out.log 2>&1
  rc=$?; echo "{\"name\":\"expert_$seed\",\"gpu\":\"$G\",\"rc\":$rc,\"wall_s\":$(( $(date +%s)-t0 ))}" >> $O/ledger.jsonl
  st=$($PY -c "import json,os;p='$out.json';print(json.load(open(p))['status'] if os.path.exists(p) else 'NO_OUTPUT')")
  echo "$(date +%T) seed $seed $st" >> $L
done
echo "$(date +%T) shard $K done" >> $L
