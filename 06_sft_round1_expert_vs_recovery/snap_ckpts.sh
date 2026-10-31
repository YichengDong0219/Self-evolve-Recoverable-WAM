#!/bin/bash
# Keep every intermediate transformer of the value-off runs (Claude, 2026-10-29, user request).
# The trainer keeps only the newest checkpoint (CHECKPOINT_TOTAL_LIMIT = 1, interval 1000 + final), so each new
# save deletes the previous one ~2.6 h later. This loop copies each checkpoint's transformer/ (~10 GB, no DeepSpeed
# state) to ~/fact_sft_amax/ckpts/<tag>_step<N>/ once its weights file is complete (size equals the released
# transformer's and unchanged for 60 s). Copies are written to a .tmp dir and renamed, never overwritten.
# Single instance (flock). Exits once both arms have their step-3194 copy.
# usage: setsid nohup bash snap_ckpts.sh > ~/fact_sft_amax/snap_ckpts.out 2>&1 &
set -u
exec 9> ~/fact_sft_amax/.snap_ckpts.lock
flock -n 9 || { echo "another snap_ckpts.sh is running"; exit 1; }
R=~/fact_sft_amax/runs; K=~/fact_sft_amax/ckpts; LOG=~/fact_sft_amax/snap_ckpts.txt; FINAL=3194
REF=$(stat -c %s /workspace/FACT/models/fact-wam/transformer/diffusion_pytorch_model.bin)
mkdir -p $K
log() { echo "$(date '+%F %T') $*" >> $LOG; }
log "start (weights size $REF)"
while true; do
  for tag in mugE_nv mugR_nv; do
    for src in $R/$tag/models/checkpoint_*_step_*/transformer; do
      [ -d "$src" ] || continue
      step=$(echo "$src" | sed -E 's/.*_step_([0-9]+)\/transformer/\1/')
      dst=$K/${tag}_step$step
      [ -e $dst ] && continue
      f=$src/diffusion_pytorch_model.bin
      [ -e $f ] && [ -e $src/config.json ] || continue
      s1=$(stat -c %s $f 2>/dev/null); sleep 60; s2=$(stat -c %s $f 2>/dev/null)
      [ "$s1" = "$REF" ] && [ "$s2" = "$REF" ] || continue
      rm -rf $dst.tmp && cp -r $src $dst.tmp && mv $dst.tmp $dst \
        && log "saved $dst ($(du -sh $dst | cut -f1)); free $(df -h /newhome | tail -1 | awk '{print $4}')" \
        || log "copy FAILED for $src"
    done
  done
  [ -e $K/mugE_nv_step$FINAL ] && [ -e $K/mugR_nv_step$FINAL ] && { log "both final copies present; exit"; exit 0; }
  sleep 120
done
