#!/bin/bash
# PRM stage of one TTS round on Amax (Claude, 2026-11-22; protocol 12 step 2).
#  1. wait until the reserved card's rollout worker has finished both passes ("second pass done"), then require the
#     card to be fully idle (no compute process, < 1 GB used);
#  2. start prm_service.py (vLLM, PRM venv, offline) on that card; check every GPU process of the service tree sits
#     on that card;
#  3. smoke test (prm_smoke.py); on failure stop the service and leave PRM_SMOKE_FAILED;
#  4. run prm_features.py every 5 min until EXEC_DONE exists and every episode with exec output has PRM features
#     (or a pass after EXEC_DONE adds nothing); stop the service; touch PRM_DONE.
# usage: setsid nohup bash prm_chain_amax.sh <smi> <round dir> > /dev/null 2>&1 < /dev/null &
set -u
G=$1; RD=$2; M=/workspace/fact_local/mug_loop_1121; P=/workspace/prm
PY=/workspace/venvs/prm_dopamine/bin/python
L=$RD/prm_chain.log; S=$RD/prm_service.log
log() { echo "$(date '+%F %T') $*" >> $L; }
idle() { [ -z "$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader)" ] && \
         [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024 ]; }
tree() { local p=$1; echo $p; for c in $(ps -o pid= --ppid $p); do tree $c; done; }
stop_service() { kill -- -$SPG 2>/dev/null; sleep 10; kill -9 -- -$SPG 2>/dev/null; log "service stopped"; }
log "start: gpu $G, round $RD"
until grep -q "second pass done" $RD/progress_gpu$G.txt 2>/dev/null; do sleep 60; done
sleep 30
idle || { log "gpu $G not idle -> abort"; exit 2; }
UUID=$(nvidia-smi -i $G --query-gpu=uuid --format=csv,noheader)
cd $M/code
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 \
  setsid $PY prm_service.py --model $P/models/GRM-2.0-8B-Preview --port 8650 \
    --prompt-src $P/Robo-Dopamine/examples/inference.py --blank-goal $P/Robo-Dopamine/examples/blank_goal.png \
    > $S 2>&1 < /dev/null &
SPID=$!; sleep 2; SPG=$(ps -o pgid= -p $SPID | tr -d ' ')
echo $SPID > $RD/prm_service.pid
log "service pid $SPID pgid $SPG (uuid $UUID)"
for i in $(seq 90); do
  grep -q PRM_SERVICE_READY $S && break
  ps -p $SPID > /dev/null || { log "service died during start-up (see prm_service.log)"; touch $RD/PRM_SMOKE_FAILED; exit 1; }
  sleep 20
done
grep -q PRM_SERVICE_READY $S || { log "service not ready after 30 min"; stop_service; touch $RD/PRM_SMOKE_FAILED; exit 1; }
PIDS=" $(tree $SPID | tr '\n' ' ') "
BAD=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | tr -d ' ' | \
      awk -F, -v u=$UUID -v ps="$PIDS" 'index(ps, " "$2" ") && $1 != u')
[ -z "$BAD" ] || { log "service process on a wrong GPU: $BAD"; stop_service; touch $RD/PRM_SMOKE_FAILED; exit 1; }
log "service ready; $(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader) used on gpu $G"
$PY prm_smoke.py --round-dir $RD >> $L 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "smoke test FAILED (rc $rc)"; stop_service; touch $RD/PRM_SMOKE_FAILED; exit 1; fi
log "smoke test passed"
while true; do
  $PY prm_features.py --round-dir $RD > $RD/prm_pass.out 2>&1 || log "prm_features rc $?"
  cat $RD/prm_pass.out >> $L
  work=$(grep -c " +[1-9][0-9]* queries" $RD/prm_pass.out)
  log "pass: $work episodes updated; $(grep PRM_PASS_DONE $RD/prm_pass.out)"
  if [ -e $RD/EXEC_DONE ] && [ "$work" -eq 0 ]; then break; fi
  [ "$work" -eq 0 ] && sleep 300
done
stop_service
touch $RD/PRM_DONE
log "PRM stage done"
