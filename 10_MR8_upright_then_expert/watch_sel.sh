#!/bin/bash
# Wake-up watcher for an MR8 selection run on the cluster (Claude 2026-11-01). usage: watch_sel.sh <run name>
# Read-only; polls every 3 min; exits when the run's driver printed "done" or no driver process is left.
C=$USER@CLUSTER_HOST; R=$1
while true; do
  s=$(ssh -o BatchMode=yes -o ConnectTimeout=20 $C "cd ~/fact_local/bench_mr8; echo \$(ls $R/*/selection.json 2>/dev/null | wc -l) \$(grep -c SELECTED $R.out) \$(grep -c '^done' $R.out) \$(pgrep -u $USER -f '[s]im_select_mr8.py .*/$R ' | wc -l)" 2>/dev/null)
  set -- $s
  echo "$(date '+%F %T') $R verdicts=${1:-?} selected=${2:-?} done=${3:-?} drivers=${4:-?}"
  [ "${3:-0}" -ge 1 ] 2>/dev/null && { echo "RUN DONE"; exit 0; }
  [ "${4:-1}" = "0" ] && { echo "NO DRIVER"; exit 0; }
  sleep 180
done
