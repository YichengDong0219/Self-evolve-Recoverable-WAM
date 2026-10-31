#!/bin/bash
# Amax copy of eval_task.sh (Claude, 2026-10-29): same rollout protocol, /newhome paths, simulator env from env.sh.
# 10-29: --skip_future_state_value (value head never used at eval; verified: executed actions bit-identical on 3 seeds)
# usage: eval_task_amax.sh <gpu> <tag> <transformer_path> <seed_file> <task> <out_root>
set -u
G=$1; TAG=$2; TP=$3; SEEDS=$4; TASK=$5; P=$((8120 + G))
O=$6/$TAG; C=/workspace/fact_local/code; R=/workspace/fact_local/runs_server  # PL: env.sh sets L
mkdir -p $O; PL=$O/progress_gpu$G.txt
bash $C/start_server_amax.sh EVAL_${TAG}_gpu$G $G $P --enable_prefix_cache --skip_future_state_value --transformer_path $TP
for i in $(seq 90); do grep -q "ready and listening" $R/server_EVAL_${TAG}_gpu$G/server.log && break; sleep 10; done
grep -q "ready and listening" $R/server_EVAL_${TAG}_gpu$G/server.log || { echo "$(date +%T) server not ready" >> $PL; kill $(cat $R/server_EVAL_${TAG}_gpu$G/pid); exit 1; }
echo "$(date +%T) server ready ($TP)" >> $PL
FACT_ARCH=8.9 source /workspace/fact_local/env.sh $G
for s in $(cat $SEEDS); do
  out=$O/${TASK}_seed$s
  [ -e $out.json ] && continue
  for try in 1 2 3; do
    t0=$(date +%s)
    timeout 1500 python $C/g0_driver.py rollout --task $TASK --config demo_clean --start-seed $s --max-tries 1 \
      --port $P --out $out.json > $out.try$try.log 2>&1
    echo "{\"name\":\"eval_${TAG}_$s\",\"gpu\":\"$G\",\"rc\":$?,\"wall_s\":$(( $(date +%s)-t0 ))}" >> $O/ledger.jsonl
    [ -e $out.json ] && break
  done
  r=$(python -c "import json,os;p='$out.json';print(json.load(open(p))['success'] if os.path.exists(p) else 'INVALID')")
  echo "$(date +%T) seed $s $r" >> $PL
done
p=$(cat $R/server_EVAL_${TAG}_gpu$G/pid); kill $p; sleep 5; ps -p $p > /dev/null && kill -9 $p
echo "$(date +%T) done, server stopped" >> $PL
