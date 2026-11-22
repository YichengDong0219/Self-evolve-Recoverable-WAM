#!/bin/bash
# Gate inputs Amax -> cluster, relayed through this machine (Claude, 2026-11-22). Copies gate_pairs.json and the
# drawn held-out episodes (<ep>.json + <ep>.tts.npz) of one round; checks sha256 on both ends.
# usage: bash gate_sync_cluster.sh <round name, e.g. r0> <local scratch dir>
set -eu
RND=$1; TMP=$2/gate_sync_$RND; mkdir -p $TMP
A=$USER@${AMAX_HOST:-AMAX_HOST}; C=$USER@${CLUSTER_HOST:-CLUSTER_HOST}
AR=/workspace/fact_local/mug_loop_1121/$RND; CR=/workspace_cluster/fact_local/mug_loop_1121/$RND
scp -q $A:$AR/gate_pairs.json $TMP/
SEEDS=$(python3 -c "import json; print(' '.join(str(s) for s in sorted({x['seed'] for x in json.load(open('$TMP/gate_pairs.json'))['states']})))")
FILES=$(for s in $SEEDS; do echo hanging_mug_seed$s.json hanging_mug_seed$s.tts.npz; done | tr "\n" " ")
for f in $FILES; do scp -q $A:$AR/$f $TMP/; done
ssh -n $C "mkdir -p $CR"
scp -q $TMP/gate_pairs.json $(for f in $FILES; do echo $TMP/$f; done) $C:$CR/
SA=$(ssh -n $A "cd $AR && sha256sum gate_pairs.json $FILES" | sort -k2)
SC=$(ssh -n $C "cd $CR && sha256sum gate_pairs.json $FILES" | sort -k2)
[ "$SA" == "$SC" ] && [ $(echo "$SA" | wc -l) -eq $((2 * $(echo $SEEDS | wc -w) + 1)) ] && echo "SYNC_OK $(echo $SEEDS | wc -w) episodes, $(echo "$SA" | wc -l) files sha256-identical" || { echo "SHA MISMATCH"; exit 1; }
