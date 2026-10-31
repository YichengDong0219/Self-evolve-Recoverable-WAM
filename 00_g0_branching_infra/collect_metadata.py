"""Bounded CPU metadata collection; no tensor/model/CUDA imports."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--fact',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    inventory=[]; exported={}
    for x in sorted((a.fact/'models').rglob('*')):
        if not x.is_file(): continue
        rel=str(x.relative_to(a.fact)); size=x.stat().st_size
        inventory.append({'path':rel,'resolved':str(x.resolve()),'size':size})
        if x.suffix=='.json' and size<=2_000_000:
            raw=x.read_bytes(); target=a.out/rel; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
            exported[rel]=hashlib.sha256(raw).hexdigest()
    log=a.fact/'setup_logs/download_ranges.log'
    if log.exists() and log.stat().st_size<=5_000_000:
        raw=log.read_bytes(); (a.out/'download_ranges.log').write_bytes(raw); exported['download_ranges.log']=hashlib.sha256(raw).hexdigest()
    (a.out/'manifest.json').write_text(json.dumps({'inventory':inventory,'exported_sha256':exported,'weight_bytes_independently_hashed':False},indent=2)+'\n')
