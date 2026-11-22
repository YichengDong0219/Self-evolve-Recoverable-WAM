#!/bin/bash
# N = 32 smoke tests on Amax (Claude, 2026-11-22, PROTOCOL addendum 4): (1) n32_generate.py on 170001 decision 0 with a
# TTS server on smi 4 (candidate ids 0-7 must equal the recorded round-0 candidates); (2) continue_n32.py of 170001 d0 c0
# (recorded candidate) on a multi-connection probe server on smi 5 (action hash must equal the gate row 137abec80ba60a07).
M=/workspace/fact_local/mug_loop_1121; S=$M/r0/n32_smoke; mkdir -p $S/gen $S/cands $S/cont_amax
cd $M
bash code/start_server_tts_amax.sh n32smoke_tts 4 8704
bash code/start_server_multi.sh n32smoke_multi 5 8805 fact --enable_prefix_cache --skip_future_state_value
/workspace/miniconda3/envs/RoboTwin/bin/python -c "import numpy as np; z=np.load('r0/hanging_mug_seed170001.tts.npz'); np.savez_compressed('$S/cands/seed170001_d0.cands.npz', cands=z['cands'][0].astype(np.float32))"
wait_ready() { for i in $(seq 1 120); do grep -q "ready and listening" $1/server.log 2>/dev/null && return 0; sleep 10; done; return 1; }
wait_ready $M/runs_server/server_n32smoke_tts || echo "TTS server not ready" >> $S/smoke.log
wait_ready /workspace/fact_local/runs_server/server_n32smoke_multi || echo "multi server not ready" >> $S/smoke.log
(FACT_ARCH=8.9 source /workspace/fact_local/env.sh 4 > /dev/null; export PYTHONPATH=$PYTHONPATH:$M/code; timeout 1800 python $M/code/n32_generate.py --episode $M/r0/hanging_mug_seed170001.json --decisions 0 --port 8704 --out-dir $S/gen > $S/gen.log 2>&1) &
(FACT_ARCH=8.9 source /workspace/fact_local/env.sh 5 > /dev/null; export PYTHONPATH=$PYTHONPATH:$M/code; timeout 1800 python $M/code/continue_n32.py --episode $M/r0/hanging_mug_seed170001.json --pairs 0:0 --port 8805 --cands-dir $S/cands --out-dir $S/cont_amax > $S/cont_amax.log 2>&1) &
wait
kill -- -$(ps -o pgid= -p $(cat $M/runs_server/server_n32smoke_tts/pid) | tr -d ' ') 2>/dev/null
kill -- -$(ps -o pgid= -p $(cat /workspace/fact_local/runs_server/server_n32smoke_multi/pid) | tr -d ' ') 2>/dev/null
echo "$(date +%T) SMOKE_DONE" >> $S/smoke.log
