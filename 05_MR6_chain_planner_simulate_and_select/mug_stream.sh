#!/bin/bash
# Stream new hanging_mug FACT failures from the remote collector into local sim-and-select (Claude, 2026-10-28).
# usage: mug_stream.sh <gpu,gpu,...> [max_candidates=8]
# Every 10 min: rsync ~/fact_collect_mug/rollouts/*.json from the remote, derive the takeover step of each new failure
# (mug_takeover_step.py; unusable states are listed with step null and skipped), append them to bench/stream_jobs.jsonl
# and run sim_select.py over the whole list (finished states are skipped by their selection.json).
# Stops after a cycle in which the remote reports every pipeline done and no new job appeared.
set -u
GPUS=$1; MAXC=${2:-8}
L=/workspace/fact_local; R=$L/stream_rollouts; J=$L/bench/stream_jobs.jsonl; LOG=$L/bench/stream.log
mkdir -p $R; touch $J
PY=/workspace/miniconda3/envs/RoboTwin/bin/python
while true; do
  rsync -a user@CLUSTER_HOST:'fact_collect_mug/rollouts/*.json' $R/ 2>/dev/null
  n_new=$(cd $L/code && $PY - "$R" "$J" <<'EOF'
import glob, json, os, sys
from mug_takeover_step import takeover_step
R, J = sys.argv[1], sys.argv[2]
seen = {json.loads(l)["rollout"] for l in open(J) if l.strip()}
skip = os.path.join(os.path.dirname(J), "stream_skipped.jsonl")
seen |= {json.loads(l)["rollout"] for l in open(skip)} if os.path.exists(skip) else set()
n = 0
for f in sorted(glob.glob(os.path.join(R, "*.json"))):
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
EOF
)
  echo "$(date +%T) new jobs $n_new, total $(wc -l < $J)" >> $LOG
  [ "$n_new" -gt 0 ] && (cd $L/code && $PY -u sim_select.py $J $L/bench/SEL_stream $GPUS $MAXC 2 >> $LOG 2>&1)
  done_remote=$(ssh -o ConnectTimeout=20 user@CLUSTER_HOST 'cat ~/fact_collect_mug/progress_gpu*.txt 2>/dev/null | grep -c "pipeline done"' 2>/dev/null || echo 0)
  if [ "$done_remote" = "4" ] && [ "$n_new" = "0" ]; then echo "$(date +%T) remote done, stream finished" >> $LOG; break; fi
  sleep 600
done
