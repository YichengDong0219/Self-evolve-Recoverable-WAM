"""Shared PRM / critic helpers (Claude, 2026-11-21). Pure numpy + PIL (no torch, no vLLM), importable from the PRM
service, the offline feature extractor and the CPU critic trainer.

Image convention (pre-registered, PROTOCOL addendum): every image the PRM sees -- real or imagined -- is at FACT's
canvas view sizes: head 256x192, wrists 128x96 (real frames are resized with PIL bilinear; imagined canvases
[192, 384] are split as head = [:, :256], left wrist = [:96, 256:], right wrist = [96:, 256:]). The reference START is
the episode's first head frame; the reference END is Robo-Dopamine's blank goal image.
Action features use the aloha-agilex qpos layout: left arm 0-5, left gripper 6, right arm 7-12, right gripper 13.
"""
import numpy as np
from PIL import Image

HEAD, WRIST = (256, 192), (128, 96)          # (width, height)
LEFT_ARM, LEFT_G, RIGHT_ARM, RIGHT_G = slice(0, 6), 6, slice(7, 13), 13


def _resize(img, size):
    img = np.asarray(img, dtype=np.uint8)
    if img.shape[1] == size[0] and img.shape[0] == size[1]:
        return img
    return np.asarray(Image.fromarray(img).resize(size, resample=Image.BILINEAR), dtype=np.uint8)


def real_views(head, left, right):
    """Real 3-view frames (native resolution, HWC uint8) -> PRM-size views."""
    return [_resize(head, HEAD), _resize(left, WRIST), _resize(right, WRIST)]


def canvas_views(canvas_hwc):
    """FACT canvas [192, 384, 3] (imagined keyframe) -> [head, left, right] at PRM sizes."""
    c = np.asarray(canvas_hwc, dtype=np.uint8)
    if c.shape[:2] != (192, 384):
        raise ValueError(f"unexpected canvas shape {c.shape}")
    return [c[:, :256].copy(), c[:96, 256:].copy(), c[96:, 256:].copy()]


def imagined_canvas(imgs, n, key):
    """imgs: uint8 [N, 3, 4, H, W] from the TTS server; key: 1..4 (= +12/+24/+36/+48 steps) -> HWC canvas."""
    return np.transpose(np.asarray(imgs[n, :, key - 1]), (1, 2, 0))


def action_features(state, cand):
    """Generic chunk statistics: per-arm joint-space displacement (L2, start->end) and gripper command change."""
    s, c = np.asarray(state, dtype=np.float64), np.asarray(cand, dtype=np.float64)
    return [float(np.linalg.norm(c[-1, LEFT_ARM] - s[LEFT_ARM])), float(np.linalg.norm(c[-1, RIGHT_ARM] - s[RIGHT_ARM])),
            float(c[-1, LEFT_G] - s[LEFT_G]), float(c[-1, RIGHT_G] - s[RIGHT_G])]


Q_FEATURES = ["hop_cur_to_im48", "phi_im48", "phi_im24", "dq_left", "dq_right", "dgrip_left", "dgrip_right"]
V_FEATURES = ["phi", "hop_prev", "t_frac", "grip_left", "grip_right"]


def q_score(model, feats):
    """Linear critic: model = {mean, std, w, b} (lists), feats = [N, F] -> [N] scores."""
    x = (np.asarray(feats, dtype=np.float64) - np.asarray(model["mean"])) / np.asarray(model["std"])
    return x @ np.asarray(model["w"]) + float(model["b"])
