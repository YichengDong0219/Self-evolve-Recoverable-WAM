"""Build the TTS inference server from a SHA-pinned copy of FACT scripts/inference_server.py (Claude, 2026-11-21;
user: "开始执行"). Never edits upstream. = build_server_probe.py's five probe changes (explicit per-candidate seeds,
all candidate actions returned) plus three TTS changes:
  * every candidate's imagined keyframes are returned as probe_candidate_images, uint8 [N, 3, 4, H, W]
    (keyframes +12/+24/+36/+48 steps; the reference frame 0 is dropped; FACT canvas layout: head 256x192 on the
    left, left wrist 128x96 top-right, right wrist 128x96 bottom-right);
  * the expanded float video of the server-selected candidate ("images", ~42 MB) is then not sent;
  * nothing else changes (FACT's own value BoN choice stays in "selected_index"/"values_per_sample" as a baseline).
usage: python build_server_tts.py --source <FACT>/scripts/inference_server.py --sha <sha256> --out <dir>
       --probe-request <fact_local/code/probe_request.py>
"""
import argparse
import hashlib
from pathlib import Path

PROBE_CHANGES = [
    ('        best_of_N = max(1, int(observation.get("best_of_N", 1)))',
     '        from probe_request import request_seeds\n        best_of_N = observation.get("best_of_N", 1)\n        probe_seeds = request_seeds(observation, best_of_N)'),
    ('imgs, action_raw, future_state_raw, value_raw = _run_pipe(None)',
     'imgs, action_raw, future_state_raw, value_raw = _run_pipe(torch.Generator(device=device).manual_seed(probe_seeds[0]))'),
    ('            base_seed = int(time.time_ns() & 0xFFFFFFFF)\n', ''),
    ('generators = [torch.Generator(device=device).manual_seed(base_seed + i) for i in range(best_of_N)]',
     'generators = [torch.Generator(device=device).manual_seed(seed) for seed in probe_seeds]'),
    ('        result = {"action": action.cpu()}',
     '        result = {"action": action.cpu(), "probe_seeds": probe_seeds}\n        if best_of_N > 1:\n            result["probe_candidate_actions"] = torch.stack([s[0] for s in samples]).cpu()'),
]
TTS_CHANGES = [
    ('        if best_of_N == 1:\n',
     '        cand_imgs = None\n        if best_of_N == 1:\n'),
    ('            imgs = None if imgs_batch is None else imgs_batch[selected_index : selected_index + 1]\n',
     '            imgs = None if imgs_batch is None else imgs_batch[selected_index : selected_index + 1]\n'
     '            cand_imgs = imgs_batch\n'),
    ('            result["images"] = imgs.detach().float().cpu()\n',
     '            result["images"] = imgs.detach().float().cpu()\n'
     '        if best_of_N > 1 and cand_imgs is not None:\n'
     '            _ci = cand_imgs.detach().float()\n'
     '            if _ci.ndim == 5 and _ci.shape[2] > 1:\n'
     '                _ci = _ci[:, :, 1:]\n'
     '            result["probe_candidate_images"] = ((_ci.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu()\n'
     '            result.pop("images", None)\n'),
]


def build(source, expected_sha, output, helper):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("source SHA mismatch")
    text = raw.decode()
    for old, new in PROBE_CHANGES + TTS_CHANGES:
        if text.count(old) != 1:
            raise ValueError("patch context missing or ambiguous: " + old[:80])
        text = text.replace(old, new)
    compile(text, "inference_server_tts.py", "exec")
    output.mkdir(parents=True, exist_ok=False)
    (output / "inference_server_tts.py").write_text(text)
    (output / "probe_request.py").write_bytes(helper.read_bytes())
    print("built", output / "inference_server_tts.py", hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True); p.add_argument("--sha", required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--probe-request", type=Path, required=True)
    a = p.parse_args()
    build(a.source, a.sha, a.out, a.probe_request)
