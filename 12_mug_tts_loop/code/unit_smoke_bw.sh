#!/bin/bash
# smoke test: FACT probe server on PRO 6000 gpu 0 (fact_bw), driver (sim) on 4090 gpu 6; one pair, K=1 (Claude, 11-22)
M=/workspace/fact_local/mug_loop_1121; G=0; DG=6; PORT=8400; V=unit_smoke_bw_gpu0
OUT=$M/r0/unit_smoke_bw; mkdir -p $OUT; L=$OUT/smoke.log
log() { echo "$(date '+%F %T') $*" >> $L; }
[ -z "$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader)" ] || { log "gpu $G busy"; exit 1; }
bash $M/code/start_server_bw.sh $V $G $PORT --enable_prefix_cache --skip_future_state_value
SD=/workspace/fact_local/runs_server/server_$V
for i in $(seq 120); do grep -q "ready and listening" $SD/server.log 2>/dev/null && break; ps -p $(cat $SD/pid) >/dev/null || break; sleep 10; done
log "server ready: $(grep -c 'ready and listening' $SD/server.log); mem $(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader)"
( FACT_ARCH=8.9 source /workspace/fact_local/env.sh $DG > /dev/null; export PYTHONPATH=$PYTHONPATH:$M/code
  timeout 1800 python $M/code/continue_record.py --episode $M/r0/hanging_mug_seed170001.json --pairs 0:0 --port $PORT --unit 1 --out-dir $OUT ) >> $OUT/driver.log 2>&1
log "K=1 rc $?; server gpu mem $(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader)"
pid=$(cat $SD/pid); kill -- -$(ps -o pgid= -p $pid | tr -d ' ') 2>/dev/null; sleep 5; kill -9 $pid 2>/dev/null
log "server stopped"; touch $OUT/SMOKE_DONE
