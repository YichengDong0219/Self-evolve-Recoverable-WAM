#!/bin/bash
# One TTS rollout worker on one fully idle Amax 4090 (Claude, 2026-11-21): starts a TTS server, runs tts_driver.py over
# its share of seeds (seed list file, one seed per line), skips seeds whose output exists, stops the server.
# usage: run_worker_amax.sh <smi index> <seed file> <out dir> <random|critic> [critic host:port]
set -u
G=$1; SEEDS=$2; O=$3; SEL=$4; CA=${5:-127.0.0.1:8650}; P=$((8200 + G))
M=/workspace/fact_local/mug_loop_1121; R=$M/runs_server; V=tts_${SEL}_gpu$G
mkdir -p $O; PL=$O/progress_gpu$G.txt
exec 9>$O/worker_gpu$G.lock; flock -n 9 || { echo "worker lock held"; exit 1; }
[ -z "$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader)" ] && \
  [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024 ] || { echo "$(date +%T) gpu $G not idle" >> $PL; exit 2; }
bash $M/code/start_server_tts_amax.sh $V $G $P
for i in $(seq 120); do grep -q "ready and listening" $R/server_$V/server.log 2>/dev/null && break; sleep 10; done
grep -q "ready and listening" $R/server_$V/server.log || { echo "$(date +%T) server not ready" >> $PL; kill $(cat $R/server_$V/pid); exit 1; }
echo "$(date +%T) server ready" >> $PL
FACT_ARCH=8.9 source /workspace/fact_local/env.sh $G
export PYTHONPATH=$PYTHONPATH:$M/code
for s in $(cat $SEEDS); do
  out=$O/hanging_mug_seed$s
  [ -e $out.json ] && continue
  [ -e $out.invalid ] && continue
  t0=$(date +%s)
  timeout 2400 python $M/code/tts_driver.py --start-seed $s --max-tries 1 --port $P --select $SEL --critic-addr $CA \
    --out $out.json > $out.log 2>&1
  rc=$?
  if [ ! -e $out.json ]; then grep -q "no expert-valid seed" $out.log && touch $out.invalid; fi
  echo "{\"seed\":$s,\"gpu\":$G,\"rc\":$rc,\"wall_s\":$(( $(date +%s)-t0 )),\"ok\":$([ -e $out.json ] && echo true || echo false)}" >> $O/ledger.jsonl
  echo "$(date +%T) seed $s rc $rc $([ -e $out.json ] && python -c "import json;d=json.load(open('$out.json'));print(d['success'],d['steps'])" || echo NOJSON)" >> $PL
done
p=$(cat $R/server_$V/pid); kill $p; sleep 5; ps -p $p > /dev/null && kill -9 $p
echo "$(date +%T) done, server stopped" >> $PL
