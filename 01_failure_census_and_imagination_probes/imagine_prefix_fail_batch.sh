#!/bin/bash
# Reconstruct FACT imagination up to the takeover step for every executed takeover attempt that FAILED (follow-up to imagine_prefix_batch.sh) (Claude, 2026-10-27).
# usage: imagine_prefix_batch.sh <server_gpu> <sim_gpu>   (port 8095, server variant IMR2; stopped at the end)
set -u
SG=$1; G=$2; P=8095; R=~/fact_g0_claude/runs; C=~/fact_g0_claude/code; O=~/fact_takeover/imagine_prefix; mkdir -p $O
bash $C/start_server.sh IMR2 $SG $P --return_images
for i in $(seq 90); do grep -q "ready and listening" $R/server_IMR2/server.log && break; sleep 10; done
grep -q "ready and listening" $R/server_IMR2/server.log || { echo "server not ready" >> $O/progress_fail.txt; kill $(cat $R/server_IMR2/pid); exit 1; }
cd /workspace_cluster/RoboTwin-FACT
export PATH=/workspace_cluster/miniconda3/envs/robotwin/bin:/workspace_cluster/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 TORCH_EXTENSIONS_DIR=/workspace_cluster/.cache/openwam-robotwin-torch-extensions
export CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 FACT_PROBE_DIR=$C PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/RoboTwin-FACT:$C
python3 - > $O/jobs_fail.txt <<'EOF'
import glob, json, os
seen = set()
for j in sorted(glob.glob("/workspace_cluster/fact_takeover/*/*.json")):
    d = json.load(open(j))
    if ("takeover" in j or "dispatch" in j) and d.get("status") != "not_applicable" and os.path.exists(j[:-5] + ".mp4"):
        seen.add((d["rollout"], d["branch_step"]))
for r, k in sorted(seen):
    print(r, k)
EOF
while read -r roll k; do
  out=$O/$(basename $roll .json)_k$k
  [ -e $out.json ] && continue
  t0=$(date +%s)
  timeout 1800 python $C/imagine_prefix.py --rollout $roll --stop-step $k --port $P --out $out > $out.log 2>&1
  rc=$?; echo "{\"name\":\"imr_$(basename $out)\",\"gpu\":\"$SG+$G\",\"rc\":$rc,\"wall_s\":$(( $(date +%s)-t0 ))}" >> ~/fact_takeover/ledger.jsonl
  echo "$(date +%T) $(basename $out) rc=$rc" >> $O/progress_fail.txt
done < $O/jobs_fail.txt
p=$(cat $R/server_IMR2/pid); kill $p; sleep 5; ps -p $p >/dev/null && kill -9 $p
echo "$(date +%T) server stopped" >> $O/progress_fail.txt
python $C/make_recovery_imagination_videos.py --status fail >> $O/progress_fail.txt 2>&1
echo ALL_DONE >> $O/progress_fail.txt
