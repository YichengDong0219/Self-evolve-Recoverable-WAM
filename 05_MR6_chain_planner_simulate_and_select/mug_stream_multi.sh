#!/bin/bash
# Multi-machine stream of hanging_mug failures into sim-and-select (Claude, 2026-10-28). Run on the local 3090 box.
# usage: mug_stream_multi.sh <local_workers> <amax_workers> <local_shards> <amax_shards> [max_candidates=8] [repeats=2]
#   (sim_select claims states with .claim files, so extra instances on one machine can join the same shards)
#   workers: GPU list, repeat a GPU for several workers (e.g. 7,7,7); shards: "k1,k2/n" of the job list
# Each 10-min cycle: rsync new FACT rollouts from the collector (CLUSTER_HOST) -> derive takeover jobs
# (mug_takeover_step.py) -> push rollouts + job list to the Amax server (AMAX_HOST, same /newhome paths) and
# start its shard 1/2 if idle -> run local shard 0/2 -> pull Amax results (disjoint per-state dirs, per-host summary).
# Stops after a cycle in which the collector reports all pipelines done, no new job appeared and both shards are idle.
set -u
LG=$1; AG=$2; LS=$3; AS=$4; MAXC=${5:-8}; REP=${6:-2}
L=/workspace/fact_local; R=$L/stream_rollouts; J=$L/bench/stream_jobs_v7.jsonl; LOG=$L/bench/stream_v7.log; OUT=$L/bench/SEL_v7
A=user@AMAX_HOST; COL=user@CLUSTER_HOST
mkdir -p $R; touch $J
PY=/workspace/miniconda3/envs/RoboTwin/bin/python
while true; do
  rsync -a $COL:'fact_collect_mug/rollouts/*.json' $R/ 2>/dev/null
  n_new=$(cd $L/code && $PY - "$R" "$J" <<'PYEOF'
import glob, json, os, sys
from mug_takeover_step import takeover_step
R, J = sys.argv[1], sys.argv[2]
seen = {json.loads(l)["rollout"] for l in open(J) if l.strip()}
skip = os.path.join(os.path.dirname(J), "stream_skipped_v7.jsonl")
seen |= {json.loads(l)["rollout"] for l in open(skip)} if os.path.exists(skip) else set()
n = 0
L = os.path.dirname(os.path.dirname(J))
for f in sorted(glob.glob(os.path.join(L, "runs", "P1_hanging_mug_*.json")) + glob.glob(os.path.join(L, "runs_img", "IMG_hanging_mug_*.json"))
                + glob.glob(os.path.join(R, "*.json"))):
    if f in seen:
        continue
    d = json.load(open(f))
    if d["success"]:
        open(skip, "a").write(json.dumps({"rollout": f, "why": "policy_success"}) + "\n")
        continue
    k, why = takeover_step(d)
    if k is None:
        open(skip, "a").write(json.dumps({"rollout": f, "why": why}) + "\n")
        continue
    open(J, "a").write(json.dumps({"rollout": f, "step": k, "plan": "MR6", "group": "mugS", "onset": why}) + "\n")
    n += 1
print(n)
PYEOF
)
  echo "$(date +%T) new jobs $n_new, total $(wc -l < $J)" >> $LOG
  rsync -a $R/ $A:fact_local/stream_rollouts/ && rsync -a $J $A:fact_local/bench/ && rsync -a $L/code/ $A:fact_local/code/
  # finished states (selection.json) go along so the Amax shard does not redo states processed here before sharding
  rsync -a --include='*/' --include='selection.json' --exclude='*' $OUT/ $A:fact_local/bench/SEL_v7/ 2>/dev/null
  amax_busy=$(ssh -o ConnectTimeout=20 $A 'pgrep -f "[s]im_select.py /workspace/fact_local/bench/stream_jobs_v7" | wc -l' 2>/dev/null || echo 1)
  if [ "$amax_busy" = "0" ]; then
    ssh -n -f -o ConnectTimeout=20 $A "cd $L/code && FACT_ARCH=8.9 setsid nohup $PY -u sim_select.py $J $OUT $AG $MAXC $REP $AS >> $L/bench/stream_amax_v7.log 2>&1 < /dev/null &" && echo "$(date +%T) amax shards $AS started on $AG" >> $LOG
  fi
  if [ "$LG" != "none" ]; then  # "none": no free local GPU (other users), the local box only syncs
    (cd $L/code && $PY -u sim_select.py $J $OUT $LG $MAXC $REP $LS >> $LOG 2>&1)
  fi
  rsync -a $A:fact_local/bench/SEL_v7/ $OUT/ 2>/dev/null
  done_remote=$(ssh -o ConnectTimeout=20 $COL 'cat ~/fact_collect_mug/progress_gpu*.txt 2>/dev/null | grep -c "pipeline done"' 2>/dev/null || echo 0)
  amax_busy=$(ssh -o ConnectTimeout=20 $A 'pgrep -f "[s]im_select.py /workspace/fact_local/bench/stream_jobs_v7" | wc -l' 2>/dev/null || echo 1)
  if [ "$done_remote" = "4" ] && [ "$n_new" = "0" ] && [ "$amax_busy" = "0" ]; then echo "$(date +%T) all done" >> $LOG; break; fi
  sleep 600
done
