#!/bin/bash
# N = 32 smoke test on the cluster (Claude, 2026-11-22, PROTOCOL addendum 4): continue_n32.py of 170001 d0 c0 (recorded
# candidate) on a multi-connection probe server on smi 0 (action hash must equal the gate row 137abec80ba60a07).
M=/workspace_cluster/fact_local/mug_loop_1121; S=$M/r0/n32_smoke; mkdir -p $S/cands $S/cont_cluster
cd $M
bash code/start_server_multi_cluster.sh n32smoke_multi 0 8800 --enable_prefix_cache --skip_future_state_value
/workspace_cluster/miniconda3/envs/robotwin/bin/python -c "import numpy as np; z=np.load('r0/hanging_mug_seed170001.tts.npz'); np.savez_compressed('$S/cands/seed170001_d0.cands.npz', cands=z['cands'][0].astype(np.float32))"
for i in $(seq 1 120); do grep -q "ready and listening" $M/runs_server/server_n32smoke_multi/server.log 2>/dev/null && break; sleep 10; done
(FACT_ARCH=8.9 source /workspace_cluster/fact_local/env.sh 0 > /dev/null; export PYTHONPATH=$PYTHONPATH:$M/code; timeout 1800 python $M/code/continue_n32.py --episode $M/r0/hanging_mug_seed170001.json --pairs 0:0 --port 8800 --cands-dir $S/cands --out-dir $S/cont_cluster > $S/cont_cluster.log 2>&1)
kill -- -$(ps -o pgid= -p $(cat $M/runs_server/server_n32smoke_multi/pid) | tr -d ' ') 2>/dev/null
echo "$(date +%T) SMOKE_DONE" >> $S/smoke.log
