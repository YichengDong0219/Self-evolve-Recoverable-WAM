#!/bin/bash
# FACT baseline on the exact OpenWAM G0 scenes (demo_randomized, same seeds, same instruction text). Claude 2026-10-27.
set -u
D=~/fact_g0_claude/code; R=~/fact_g0_claude/runs; L=$R/CMP_progress.txt
python3 - > $R/cmp_jobs.txt <<'PY'
import json
for t in ["hanging_mug", "turn_switch"]:
    r = json.load(open(f"/workspace_cluster/OpenWAM/outputs/probe/g0_pilot_remaining_codex_20261013/{t}/results.json"))
    for e in r["episodes"]:
        print(json.dumps({"task": t, "seed": e["seed"], "instruction": e["instruction"], "openwam_success": e["success"]}))
PY
mv $R/server_A_cache_on $R/server_A_cache_on_run2 2>/dev/null
bash $D/start_server.sh A_cache_on 0 8093 --enable_prefix_cache
for i in $(seq 90); do grep -q "ready and listening" $R/server_A_cache_on/server.log && break; sleep 10; done
echo "$(date +%T) server ready" >> $L
while read -r job; do
  t=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['task'])" "$job")
  s=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['seed'])" "$job")
  ins=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['instruction'])" "$job")
  for try in 1 2 3; do  # expert check on the same seed is not perfectly reproducible; retry the SAME seed only
    bash $D/run_driver.sh 1 CMP_${t}_${s} rollout --task $t --config demo_randomized --start-seed $s --max-tries 1 --instruction "$ins" > /dev/null
    [ -e $R/CMP_${t}_${s}.json ] && break
    mv $R/CMP_${t}_${s}.log $R/CMP_${t}_${s}.try$try.log
  done
  python3 -c "import json,os;p='$R/CMP_${t}_${s}.json';d=json.load(open(p)) if os.path.exists(p) else None;print('$t',$s,'FACT',d and d['success'],d and d['steps'])" >> $L
done < $R/cmp_jobs.txt
p=$(cat $R/server_A_cache_on/pid); kill $p; sleep 5; date +%s > $R/server_A_cache_on/stop_unix
echo "$(date +%T) rollouts done, server stopped" >> $L
for n in $(ls $R/CMP_*.json | xargs -n1 basename | sed 's/.json//'); do bash $D/run_render.sh 1 $n > /dev/null; done
echo "$(date +%T) all done" >> $L
