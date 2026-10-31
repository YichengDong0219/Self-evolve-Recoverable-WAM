#!/bin/bash
# Retry expert demos that failed (curobo planning is not perfectly reproducible on a seed) (Claude, 2026-10-28).
# usage: expert_retry.sh <gpu> <shard> <n_shards>   -> new stems ..._expert_retry1 / _retry2; originals are kept.
set -u
G=$1; K=$2; N=$3
O=~/fact_collect_switch_expert; C=~/fact_g0_claude/code; L=$O/retry_gpu$G.txt
cd /workspace_cluster/RoboTwin-FACT
export PATH=/workspace_cluster/miniconda3/envs/robotwin/bin:/workspace_cluster/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_EXTENSIONS_DIR=/workspace_cluster/.cache/openwam-robotwin-torch-extensions
export CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 FACT_PROBE_DIR=$C PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/RoboTwin-FACT:$C
PY=/workspace_cluster/miniconda3/envs/robotwin/bin/python
$PY - $K $N <<'EOF' > $O/retry_list_gpu$G.txt
import json, os, sys
k, n = int(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(l) for l in open(os.path.expanduser("~/fact_collect_switch/MANIFEST_turn_switch_recovery_v1.jsonl"))]
for i, r in enumerate(rows):
    p = os.path.expanduser(f"~/fact_collect_switch_expert/demos/turn_switch_seed{r['env_seed']}_expert.json")
    ok = os.path.exists(p) and json.load(open(p))["status"] == "SUCCESS"
    if not ok and i % n == k:
        print(r["env_seed"], r["rollout"])
EOF
while read -r seed roll; do
  for t in 1 2; do
    out=$O/demos/turn_switch_seed${seed}_expert_retry$t
    [ -e $out.json ] || timeout 900 $PY $C/collect_expert_demo.py --rollout ~/fact_collect_switch/$roll --out $out > $out.log 2>&1
    st=$($PY -c "import json,os;p='$out.json';print(json.load(open(p))['status'] if os.path.exists(p) else 'NO_OUTPUT')")
    echo "$(date +%T) seed $seed retry$t $st" >> $L
    [ "$st" = "SUCCESS" ] && break
  done
done < $O/retry_list_gpu$G.txt
echo "$(date +%T) retry shard $K done" >> $L
