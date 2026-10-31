#!/bin/bash
# Run one shard of a takeover-benchmark job list on one local GPU (Claude, 2026-10-28).
# usage: bench_runner.sh <gpu> <shard> <n_shards> <jobs.jsonl> <variant> [extra takeover_bench.py args...]
set -u
G=$1; K=$2; N=$3; J=$4; V=$5; shift 5
source /workspace/fact_local/env.sh $G
O=$L/bench/${BENCH_TAG:-$V}; mkdir -p $O/logs; P=$O/progress_gpu$G.txt  # BENCH_TAG: output dir name (default = variant)
python - "$J" $K $N <<'EOF' > $O/shard_gpu$G.txt
import json, os, sys
j, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for i, line in enumerate(open(j)):
    if i % n == k:
        d = json.loads(line)
        seed = json.load(open(d["rollout"]))["env_seed"]
        print(d["rollout"], d["step"], d["plan"], f'{d["group"]}_{seed}_k{d["step"]}_{d["plan"]}')
EOF
while read -r roll step plan name; do
  [ -e $O/$name.json ] && continue
  t0=$(date +%s)
  timeout 900 python $L/code/takeover_bench.py --rollout $roll --step $step --plan $plan --variant $V --out $O/$name.json "$@" > $O/logs/$name.log 2>&1
  rc=$?
  r=$(grep -a -o '{"task".*' $O/logs/$name.log | tail -1)
  echo "$(date +%T) rc=$rc wall=$(( $(date +%s)-t0 )) $name $r" >> $P
done < $O/shard_gpu$G.txt
echo "$(date +%T) shard $K done" >> $P
