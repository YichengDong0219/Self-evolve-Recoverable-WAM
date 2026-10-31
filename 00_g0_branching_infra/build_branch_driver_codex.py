"""Build Codex extension from SHA-pinned Claude driver; never mutate author source."""
import argparse,hashlib
from pathlib import Path

def build(source,sha,out):
    raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha: raise ValueError('source SHA mismatch')
    s=raw.decode()
    edits=[
      ('    a = ap.parse_args()', '    ap.add_argument("--stream-repeat", type=int, default=0)\n    a = ap.parse_args()\n    if not 0 <= a.stream_repeat <= 8:\n        raise SystemExit("stream repeat must be 0..8")\n    if a.mode == "branch" and a.max_tries != 1:\n        raise SystemExit("branch requires --max-tries 1; no replacement scenes")'),
      ('    actions = src["executed_actions"] if src else []', '    actions = src["executed_actions"] if src else []\n    if src:\n        if hashlib.sha256(json.dumps(actions).encode()).hexdigest() != src["executed_actions_sha256"]:\n            raise SystemExit("source action SHA mismatch")\n        if a.root_seed != src["root_seed"]:\n            raise SystemExit("root seed must match original; change stream-repeat only")'),
      ('policy.bind_probe(start_id, a.root_seed, server_sink(io_log, a.save_request))', 'policy.bind_probe(start_id, a.root_seed, server_sink(io_log, a.save_request), repeat=a.stream_repeat)'),
      ('        obs = env.get_obs()', '        if a.mode == "branch" and step == stop_replay:\n            expected = src["states"].get(str(step))\n            if expected is None or capture(env) != expected:\n                raise RuntimeError("prefix state mismatch; no suffix model request")\n            if executed != actions[:stop_replay]:\n                raise RuntimeError("prefix actions mismatch")\n        obs = env.get_obs()'),
      ('"root_seed": a.root_seed, "start_id": start_id, "branch_step": stop_replay,', '"root_seed": a.root_seed, "stream_repeat": a.stream_repeat, "source_sha256": hashlib.sha256(a.actions_from.read_bytes()).hexdigest() if src else None, "start_id": start_id, "branch_step": stop_replay,')]
    for old,new in edits:
        if s.count(old)!=1: raise ValueError('ambiguous patch context')
        s=s.replace(old,new)
    compile(s,str(out),'exec')
    with out.open('x') as f:f.write(s)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--sha',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.source,a.sha,a.out)
