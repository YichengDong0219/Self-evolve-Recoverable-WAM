"""Build a separate server entry from a SHA-pinned source, never edit upstream."""
import argparse
import hashlib
from pathlib import Path


def build(source, expected_sha, output, helper):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('source SHA mismatch')
    text = raw.decode()
    changes = [
        ('        best_of_N = max(1, int(observation.get("best_of_N", 1)))',
         '        from probe_request import request_seeds\n        best_of_N = observation.get("best_of_N", 1)\n        probe_seeds = request_seeds(observation, best_of_N)'),
        ('imgs, action_raw, future_state_raw, value_raw = _run_pipe(None)',
         'imgs, action_raw, future_state_raw, value_raw = _run_pipe(torch.Generator(device=device).manual_seed(probe_seeds[0]))'),
        ('            base_seed = int(time.time_ns() & 0xFFFFFFFF)\n', ''),
        ('generators = [torch.Generator(device=device).manual_seed(base_seed + i) for i in range(best_of_N)]',
         'generators = [torch.Generator(device=device).manual_seed(seed) for seed in probe_seeds]'),
        ('        result = {"action": action.cpu()}',
         '        result = {"action": action.cpu(), "probe_seeds": probe_seeds}\n        if best_of_N > 1:\n            result["probe_candidate_actions"] = torch.stack([s[0] for s in samples]).cpu()')]
    for old, new in changes:
        if text.count(old) != 1:
            raise ValueError('patch context missing or ambiguous: ' + old)
        text = text.replace(old, new)
    compile(text, 'inference_server_probe.py', 'exec')
    output.mkdir(parents=True, exist_ok=False)
    (output/'inference_server_probe.py').write_text(text)
    (output/'probe_request.py').write_bytes(helper.read_bytes())

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True)
    p.add_argument('--sha',required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); build(a.source,a.sha,a.out,Path(__file__).with_name('probe_request.py'))
