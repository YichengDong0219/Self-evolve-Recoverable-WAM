#!/bin/bash
# one null replicate: synthetic round with no signal, gate selection, emulated continuations, critic gate
s=$1; PY=/workspace/miniconda3/envs/fact/bin/python; C=/workspace/fact_local/mug_loop_1121/code
D=nullrep/r$s; rm -rf $D; mkdir -p nullrep
$PY -I synth_round.py $D 0.0 $s > /dev/null
$PY $C/gate_select.py --round-dir $D --scan-start 170000 > /dev/null
$PY -I -c "
import json
from pathlib import Path
D=Path('$D'); g=json.load(open(D/'gate_pairs.json'))
for s,pairs in g['pairs'].items():
    want={tuple(map(int,p.split(':'))) for p in pairs.split(',')}
    rows=[json.loads(l) for l in open(D/f'hanging_mug_seed{s}.cont_all.jsonl') if l.strip()]
    (D/f'hanging_mug_seed{s}.cont.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if (r['decision'],r['cand']) in want))
"
$PY $C/critic_train.py --round-dir $D --scan-start 170000 --mode gate --out $D/gate.json > /dev/null
python3 -c "
import json; r=json.load(open('$D/gate.json')); print($s, r['gate_pass'], round(r['gate_gain'],3), [round(x,3) for x in r['gate_gain_ci']])"
