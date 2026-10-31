#!/bin/bash
# Recovery arm R189 end to end, aligned with the expert arm E189 (Claude, 2026-10-28):
# same 189 seeds (lists/R189.jsonl = expert segments of the recovery trajectories, npz key "expert"), same converter,
# same T5 / VAE caches, same config (fact_sft_configs/turn_switch_sft.py), 1000 steps, ckpt every 500, no offload;
# then post_train_eval.sh with also_base=0: R189 on GPUs 2,3 over seeds_half0/1 (130000-130119), eval_switch.sh as E189.
set -u
L=~/fact_sft/r189_pipeline.txt; C=~/fact_g0_claude/code; T=~/fact_sft/data/R189/Clean/turn_switch
log() { echo "$(date +%T) $*" >> $L; }
cd ~/fact_sft
log "convert"
/workspace_cluster/miniconda3/envs/fact/bin/python $C/npz_to_lerobot.py --list lists/R189.jsonl --out $T > convert_R189.log 2>&1 || { log "convert FAILED"; exit 1; }
log "$(tail -1 convert_R189.log)"
cd ~/FACT
export PATH=/workspace_cluster/miniconda3/envs/fact/bin:/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace_cluster/FACT:/workspace_cluster/fact_sft
log "t5"
CUDA_VISIBLE_DEVICES=0 python -m scripts.compute_t5_embedding --root $T --wan_path models/Wan2.2-TI2V-5B-Diffusers --device cuda > ~/fact_sft/t5_R189.log 2>&1 || { log "t5 FAILED"; exit 1; }
log "t5 $(ls $T/t5_embedding | wc -l) files"
log "vae"
FACT_PROJECT_DIR=~/fact_sft/runs/_vae_tmp python -m scripts.compute_vae_latents --config_module fact_sft_configs.turn_switch_sft \
  --robotwin_root ~/fact_sft/data/R189 --devices cuda:0,cuda:1,cuda:2,cuda:3 > ~/fact_sft/vae_R189.log 2>&1 || { log "vae FAILED"; exit 1; }
log "$(grep -a 'Done in' ~/fact_sft/vae_R189.log | tail -1)"
log "train"
FACT_CKPT_INTERVAL=500 FACT_DS_OFFLOAD=0 bash $C/train_sft.sh R189 ~/fact_sft/data/R189 1000 >> $L 2>&1
log "train finished: $(tail -1 ~/fact_sft/ledger.jsonl)"
bash $C/post_train_eval.sh R189 1000 500 0 > ~/fact_sft/post_R189.log 2>&1
log "post_train_eval finished"
