"""CPU-only, bounded text-source export. No model imports, GPU query or repo edits.
Run under timeout 180s; output must be a new directory. Does NOT attest weights.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

EXT = {'.py', '.sh', '.yaml', '.yml', '.json', '.toml', '.md', '.txt', '.ini', '.cfg'}

def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], timeout=20)


def collect(roots, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {'format': 'fact-source-export/1', 'repositories': {}, 'files': {}, 'skipped': []}
    total = 0
    for label, root in roots.items():
        root = root.resolve(strict=True)
        head = git(root, 'rev-parse', 'HEAD').decode().strip()
        status = git(root, 'status', '--porcelain=v1').decode()
        report['repositories'][label] = {'root': str(root), 'head': head, 'status_before': status}
        names = set(git(root, 'ls-files', '-z').decode().split('\0')) - {''}
        # These deployment files may be untracked. Do not export unrelated untracked files.
        names.update(n for n in ['BASELINE_4090.md', 'local/run_baseline.sh'] if (root/n).exists())
        for name in sorted(names):
            rel = Path(name); src = root / rel
            key = label + '/' + name
            if rel.is_absolute() or '..' in rel.parts or src.is_symlink() or not src.is_file() or src.suffix not in EXT:
                report['skipped'].append(key); continue
            if not src.resolve().is_relative_to(root) or src.stat().st_size > 2_000_000:
                report['skipped'].append(key); continue
            raw = src.read_bytes()
            if b'\0' in raw:
                report['skipped'].append(key); continue
            raw.decode('utf-8')
            total += len(raw)
            if total > 100_000_000:
                raise RuntimeError('100 MB text export cap exceeded; partial export is not complete')
            dst = output / key; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(raw)
            report['files'][key] = {'sha256': hashlib.sha256(raw).hexdigest(), 'size': len(raw)}
        # Detect source edits during capture, including assume-unchanged tracked files.
        for key, meta in report['files'].items():
            if key.startswith(label + '/'):
                if hashlib.sha256((root/key[len(label)+1:]).read_bytes()).hexdigest() != meta['sha256']:
                    raise RuntimeError('source changed during export: ' + key)
        if git(root, 'rev-parse', 'HEAD').decode().strip() != head or git(root, 'status', '--porcelain=v1').decode() != status:
            raise RuntimeError('repository changed during export')
    report['complete'] = True
    (output/'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--fact', type=Path, required=True)
    p.add_argument('--robotwin', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); r = collect({'FACT': a.fact, 'RoboTwin-FACT': a.robotwin}, a.out)
    print(json.dumps({'complete': True, 'files': len(r['files'])}))
