#!/bin/bash
# Stage-aware recovery smoke on all P1 natural failures of turn_switch and move_stapler_pad (Claude, 2026-10-27).
# hanging_mug uses the results already in ~/fact_takeover/dispatch (same plans L1/R1/R2).
# Plans tried in priority order per failure; stop at the first SUCCESS. Output ~/fact_takeover/dispatch_multi (new).
set -u
O=~/fact_takeover/dispatch_multi; mkdir -p $O; G=${1:-1}
cd /workspace_cluster/RoboTwin-FACT
export PATH=/workspace_cluster/miniconda3/envs/robotwin/bin:/workspace_cluster/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_EXTENSIONS_DIR=/workspace_cluster/.cache/openwam-robotwin-torch-extensions
export CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 FACT_PROBE_DIR=/workspace_cluster/fact_g0_claude/code
export PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/RoboTwin-FACT:/workspace_cluster/fact_g0_claude/code
R=/workspace_cluster/fact_g0_claude/runs
{
  for x in "00 96" "02 96" "03 96" "04 96" "06 96" "08 96" "09 96" "13 288" "18 96"; do
    set -- $x; echo "$R/P1_turn_switch_demo_clean_$1.json $2 W1 W2"; done
  for i in 04 07 10 16; do echo "$R/P1_move_stapler_pad_demo_clean_$i.json 192 S1 S2"; done
} > $O/jobs.txt
while read -r roll step plans; do
  n=$(basename $roll .json)
  for p in $plans; do
    out=$O/${n}_step${step}_dispatch${p}
    if [ ! -e $out.json ]; then
      t0=$(date +%s)
      timeout 900 /workspace_cluster/miniconda3/envs/robotwin/bin/python /workspace_cluster/fact_g0_claude/code/recovery_dispatch_multi.py \
        --rollout $roll --branch-step $step --plan $p --out $out > $out.log 2>&1
      echo "{\"name\":\"$(basename $out)\",\"gpu\":\"$G\",\"rc\":$?,\"wall_s\":$(( $(date +%s)-t0 ))}" >> $O/../ledger.jsonl
    fi
    st=$(python3 -c "import json,os;p='$out.json';print(json.load(open(p))['status'] if os.path.exists(p) else 'NO_OUTPUT')")
    echo "$(date +%T) $n $p $st" >> $O/progress.txt
    [ "$st" = "SUCCESS" ] && break
  done
done < $O/jobs.txt
echo ALL_DONE >> $O/progress.txt
