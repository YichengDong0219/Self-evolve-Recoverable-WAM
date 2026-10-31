"""Write the turn_switch SFT config as an edited copy of FACT's official RoboTwin config (Claude, 2026-10-28).

usage: python make_sft_config.py <FACT repo>/world_action_model/configs/robotwin.py <out.py>
Only paths, hardware, batch, steps and learning rates change; every edit must match exactly once.
"""
import sys

HEADER = ('"""\nturn_switch SFT config for the expert-demo (E200) vs recovery (R202) comparison (Claude, 2026-10-28).\n'
          'Edited copy of FACT world_action_model/configs/robotwin.py made by make_sft_config.py: only paths, hardware,\n'
          'batch, steps and LR differ. Both arms use this file and differ only in ROBOTWIN_ROOT / FACT_PROJECT_DIR.\n\n')

EDITS = [
    ('"""\nFACT (World-Action Model) — Multi-GPU RoboTwin Training Config', HEADER + 'FACT (World-Action Model) — Multi-GPU RoboTwin Training Config'),
    ('"./models/Wan2.2-TI2V-5B-Diffusers"', '"/workspace_cluster/FACT/models/Wan2.2-TI2V-5B-Diffusers"'),
    ('"./artifacts/robotwin/norm_stats_delta.json"', '"/workspace_cluster/FACT/models/fact-wam/norm_stats_delta.json"'),
    ('CHECKPOINT = None', 'CHECKPOINT = _os.environ.get("FACT_INIT_CHECKPOINT", "/workspace_cluster/FACT/models/fact-wam/transformer")'),
    ('GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]', 'GPU_IDS = [int(g) for g in _os.environ.get("FACT_GPU_IDS", "0,1,2,3").split(",")]'),
    ('DEEPSPEED_OFFLOAD_OPTIMIZER = False', 'DEEPSPEED_OFFLOAD_OPTIMIZER = _os.environ.get("FACT_DS_OFFLOAD", "0") == "1"'),
    ('BATCH_SIZE_PER_GPU = 32', 'BATCH_SIZE_PER_GPU = int(_os.environ.get("FACT_BATCH_PER_GPU", "2"))'),
    ('GRADIENT_ACCUMULATION_STEPS = 1', 'GRADIENT_ACCUMULATION_STEPS = int(_os.environ.get("FACT_GRAD_ACCUM", "4"))'),
    ('MAX_STEPS = 150_000', 'MAX_STEPS = int(_os.environ.get("FACT_MAX_STEPS", "2000"))'),
    ('LR = 2e-5', 'LR = 1e-5'),
    ('ACTION_LR = 2e-4', 'ACTION_LR = 1e-4'),
    ('    warmup_steps=500,', '    warmup_steps=100,'),
    ('CHECKPOINT_INTERVAL = 1_000', 'CHECKPOINT_INTERVAL = int(_os.environ.get("FACT_CKPT_INTERVAL", "500"))'),
    ('CHECKPOINT_TOTAL_LIMIT = 1', 'CHECKPOINT_TOTAL_LIMIT = 1'),  # a DeepSpeed state is ~68 GB; keep one
    # per-dataset sampling multipliers (hanging_mug R arm: failure-prefix dataset at ~1/3 of samples), JSON env var
    ('ROBOTWIN_DATASET_MULTIPLIERS = {\n    "*": 1.0,\n}',
     'ROBOTWIN_DATASET_MULTIPLIERS = __import__("json").loads(_os.environ.get("FACT_DATASET_MULTIPLIERS", \'{"*": 1.0}\'))'),
    # Only the FACT transformer is downloaded (not Wan2.2's); build the transformer from it, as the inference server
    # does, then load the same weights again through CHECKPOINT (restores the action heads reset at construction).
    ('    pretrained=WAN_MODEL_DIR,\n    checkpoint=CHECKPOINT,',
     '    pretrained=WAN_MODEL_DIR,\n    transformer_pretrained=CHECKPOINT,\n    checkpoint=CHECKPOINT,'),
]


def main():
    src, out = sys.argv[1], sys.argv[2]
    s = open(src).read()
    for old, new in EDITS:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f"edit target found {n} times: {old!r}")
        s = s.replace(old, new)
    open(out, "x").write(s)
    print("wrote", out)


if __name__ == "__main__":
    main()
