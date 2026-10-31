#!/bin/bash
# Relay mugM_nv_step3000 Amax -> local -> cluster for line 2 (Claude 2026-10-31). Verifies size, removes the local copy.
set -e
L=/workspace/fact_local/ckpt_relay/mugM_nv_step3000; T=/workspace_cluster/fact_sft/ckpts_selfevo/mugM_nv_step3000
rsync -a user@AMAX_HOST:/workspace/fact_sft_amax/ckpts/mugM_nv_step3000/ $L/
ssh -o BatchMode=yes user@CLUSTER_HOST "mkdir -p $T"
rsync -a $L/ user@CLUSTER_HOST:$T/
a=$(sha256sum $L/diffusion_pytorch_model.bin | cut -c1-16); b=$(ssh -o BatchMode=yes user@CLUSTER_HOST "sha256sum $T/diffusion_pytorch_model.bin | cut -c1-16")
[ "$a" = "$b" ] && echo "relay ok $a" && rm -rf $L || { echo "sha mismatch $a $b"; exit 1; }
