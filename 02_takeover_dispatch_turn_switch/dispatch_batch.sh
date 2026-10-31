#!/bin/bash
# Recovery-dispatch trial (Claude, 2026-10-27): mug drop states with a verified-reachable grasp; plans in priority order,
# stop at the first SUCCESS. New output dir ~/fact_takeover/dispatch; existing smoke results untouched.
set -u
O=~/fact_takeover/dispatch; mkdir -p $O; G=1
cd /workspace_cluster/RoboTwin-FACT
export PATH=/workspace_cluster/miniconda3/envs/robotwin/bin:/workspace_cluster/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_EXTENSIONS_DIR=/workspace_cluster/.cache/openwam-robotwin-torch-extensions
export CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 FACT_PROBE_DIR=/workspace_cluster/fact_g0_claude/code
export PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/RoboTwin-FACT:/workspace_cluster/fact_g0_claude/code
python3 - > $O/jobs.txt <<'PY'
import json, glob
for f in sorted(glob.glob("/workspace_cluster/fact_takeover/smoke/*graspscan.json")):
    d = json.load(open(f)); r = {a: [p["contact_point"] for p in d["points"] if p["arm"] == a and p["pose_found"] and p["pre_reachable"] and p["grasp_reachable"]] for a in ("left", "right")}
    plans = (["L1"] if r["left"] else []) + (["R1", "R2"] if r["right"] else [])
    if plans: print(d["rollout"], d["branch_step"], " ".join(plans))
PY
while read -r roll step plans; do
  n=$(basename $roll .json)
  for p in $plans; do
    out=$O/${n}_step${step}_dispatch${p}
    t0=$(date +%s)
    timeout 900 /workspace_cluster/miniconda3/envs/robotwin/bin/python /workspace_cluster/fact_g0_claude/code/recovery_dispatch.py \
      --rollout $roll --branch-step $step --plan $p --out $out > $out.log 2>&1
    echo "{\"name\":\"$(basename $out)\",\"gpu\":\"$G\",\"rc\":$?,\"wall_s\":$(( $(date +%s)-t0 ))}" >> $O/../ledger.jsonl
    st=$(python3 -c "import json,os;p='$out.json';print(json.load(open(p))['status'] if os.path.exists(p) else 'NO_OUTPUT')")
    echo "$(date +%T) $n $p $st" | tee -a $O/progress.txt
    [ "$st" = "SUCCESS" ] && break
  done
done < $O/jobs.txt
echo ALL_DONE >> $O/progress.txt
