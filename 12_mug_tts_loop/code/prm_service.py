"""Robo-Dopamine GRM-2.0-8B-Preview scoring service (Claude, 2026-11-21; user: "开始执行"). Runs in the PRM venv
(vLLM 0.11) on one fully idle GPU. Prompt and message layout are copied from Robo-Dopamine examples/inference.py
(Apache-2.0, FlagOpen/Robo-Dopamine @2c714ab); decoding is greedy (temperature 0) for determinism (pre-registered).
Each query = task text + 8 images: REF START front, REF END front (blank goal), BEFORE front/left/right,
AFTER front/left/right -> integer percentage in [-100, 100] -> returned as a float in [-1, 1].
  incremental hop(s -> s')  = BEFORE s,      AFTER s'
  forward     phi(x)        = BEFORE start,  AFTER x
Protocol over multiprocessing.connection (authkey b"mug-tts-critic"):
  {"op": "prm", "items": [{"task", "start": [3 views], "before": [3], "after": [3]}, ...]}  -> {"scores", "raw"}
  {"op": "score", ...TTS decision...}  -> {"scores"} using the linear critic in --critic (round 1 only)
usage: python prm_service.py --model <dir> --port 8650 [--critic critic_q.json] [--blank-goal blank_goal.png]
"""
import argparse
import json
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prm_common import (action_features, canvas_views, imagined_canvas, q_score, real_views)  # noqa: E402

SYSTEM_PROMPT = None  # loaded from the Robo-Dopamine source at start-up (verbatim)


def load_prompt(src):
    text = Path(src).read_text()
    a = text.index('SYSTEM_PROMPT = """') + len('SYSTEM_PROMPT = """')
    return text[a:text.index('"""', a)]


class GRM:
    def __init__(self, model_path, blank_goal):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model_path, gpu_memory_utilization=0.85, max_model_len=8192, limit_mm_per_prompt={"image": 8},
                       enable_prefix_caching=True, trust_remote_code=True)
        self.sp = SamplingParams(temperature=0.0, max_tokens=24)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.max_pixels = 76800
            self.processor.image_processor.min_pixels = 12544
        self.blank = np.asarray(Image.open(blank_goal).convert("RGB").resize((256, 192), Image.BILINEAR), dtype=np.uint8)

    def _prompt(self, task):
        parts = SYSTEM_PROMPT.format(task=task).split("<image>")
        content = []
        for i, p in enumerate(parts):
            content.append({"type": "text", "text": p})
            if i < len(parts) - 1:
                content.append({"type": "image"})
        msgs = [{"role": "user", "content": content}]
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def score(self, items):
        prompts = []
        for it in items:
            ims = [it["start"][0], self.blank] + list(it["before"]) + list(it["after"])
            prompts.append({"prompt": self._prompt(it["task"]),
                            "multi_modal_data": {"image": [Image.fromarray(np.asarray(x, dtype=np.uint8)) for x in ims]}})
        outs = self.llm.generate(prompts, sampling_params=self.sp, use_tqdm=False)
        scores, raw = [], []
        for o in outs:
            t = o.outputs[0].text
            raw.append(t)
            try:
                v = float(t.split("<score>")[-1].split("</score>")[0].replace("%", "").strip())
                scores.append(max(-100.0, min(100.0, v)) / 100.0)
            except Exception:
                scores.append(float("nan"))
        return scores, raw


def critic_scores(grm, critic, msg):
    task, start, cur = msg["instruction"], real_views(*msg["start"]), real_views(*msg["cur"])
    imgs, cands, state = msg["imgs"], msg["cands"], msg["state"]
    n = len(cands)
    items = []
    for i in range(n):
        im48, im24 = canvas_views(imagined_canvas(imgs, i, 4)), canvas_views(imagined_canvas(imgs, i, 2))
        items += [dict(task=task, start=start, before=cur, after=im48),     # hop(cur -> im48)
                  dict(task=task, start=start, before=start, after=im48),   # phi(im48)
                  dict(task=task, start=start, before=start, after=im24)]   # phi(im24)
    s, _ = grm.score(items)
    s = np.asarray(s, dtype=np.float64).reshape(n, 3)
    feats = [list(s[i]) + action_features(state, cands[i]) for i in range(n)]
    feats = np.nan_to_num(np.asarray(feats), nan=0.0)
    return q_score(critic, feats).tolist(), feats.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--port", type=int, default=8650)
    ap.add_argument("--critic"); ap.add_argument("--prompt-src", required=True); ap.add_argument("--blank-goal", required=True)
    a = ap.parse_args()
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = load_prompt(a.prompt_src)
    grm = GRM(a.model, a.blank_goal)
    critic = json.loads(Path(a.critic).read_text()) if a.critic else None
    print("PRM_SERVICE_READY", a.port, "critic" if critic else "no-critic", flush=True)
    with Listener(("127.0.0.1", a.port), authkey=b"mug-tts-critic") as lis:
        while True:
            conn = lis.accept()
            try:
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    t0 = time.time()
                    try:
                        if msg["op"] == "prm":
                            sc, raw = grm.score(msg["items"])
                            conn.send({"scores": sc, "raw": raw, "s": round(time.time() - t0, 2)})
                        elif msg["op"] == "score":
                            if critic is None:
                                raise RuntimeError("no critic loaded")
                            sc, feats = critic_scores(grm, critic, msg)
                            conn.send({"scores": sc, "feats": feats, "s": round(time.time() - t0, 2)})
                        else:
                            raise RuntimeError(f"unknown op {msg['op']}")
                    except Exception as e:  # noqa: BLE001 - report to the client, keep serving
                        conn.send({"error": repr(e)})
            finally:
                conn.close()


if __name__ == "__main__":
    main()
