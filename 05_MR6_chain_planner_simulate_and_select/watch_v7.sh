#!/bin/bash
# Watchdog for the v7 hanging_mug data run (Claude, 2026-10-29). Prints one line per 15-min check; exits (printing
# the reason) when 100 trajectories are selected, a worker host stalls, or another user appears on a GPU we use.
L=/workspace/fact_local; A=user@AMAX_HOST; C=user@CLUSTER_HOST
count() {  # selected / processed in a SEL_v7 dir
  python3 - "$1" <<'EOF'
import glob, json, sys
s = [json.load(open(p)) for p in glob.glob(sys.argv[1] + "/*/selection.json")]
print(",".join(sorted(r["name"] for r in s if r.get("selected"))) or "-", ",".join(sorted(r["name"] for r in s)) or "-")
EOF
}
while true; do
  read ls lp < <(count $L/bench/SEL_v7)
  read as ap < <(ssh -o ConnectTimeout=20 $A "$(declare -f count); count $L/bench/SEL_v7" 2>/dev/null || echo "? ?")
  fails=$(ssh -o ConnectTimeout=20 $C 'wc -l < ~/fact_collect_mug/failures.txt' 2>/dev/null || echo "?")
  others_amax=$(ssh -o ConnectTimeout=20 $A 'for g in 4 5 6; do u=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | while read p; do ps -o user= -p $p 2>/dev/null; done | grep -v dongyic | sort -u | tr "\n" " "); [ -n "$u" ] && echo "gpu$g:$u"; done' 2>/dev/null)
  others_local=$(for g in 0 3 4 5 6 7; do u=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | while read p; do ps -o user= -p $p 2>/dev/null; done | grep -v dongyic | sort -u | tr "\n" " "); [ -n "$u" ] && echo "gpu$g:$u"; done)
  wa=$(ssh -o ConnectTimeout=20 $A 'pgrep -u $USER -f "[t]akeover_bench.py" | wc -l' 2>/dev/null || echo "?")
  wl=$(pgrep -u $USER -f "[t]akeover_bench.py" | wc -l)
  # local pulls the Amax results every cycle: count the union of state names, not the sum
  tot=$(echo "$ls,$as" | tr ',' '\n' | grep -v -e '^-$' -e '^?$' -e '^$' | sort -u | wc -l)
  proc=$(echo "$lp,$ap" | tr ',' '\n' | grep -v -e '^-$' -e '^?$' -e '^$' | sort -u | wc -l)
  echo "$(date +%T) selected $tot / processed $proc | failures collected $fails | workers local $wl amax $wa | others: ${others_amax}${others_local}"
  if [ "$tot" -ge 100 ]; then echo "TARGET 100 reached"; exit 0; fi
  if [ -n "$others_amax$others_local" ]; then echo "OTHER USER ON OUR GPU"; exit 0; fi
  [ "$wa" = "0" ] && idle=$((${idle:-0}+1)) || idle=0  # Amax sim_select restarts each stream cycle: exit only after 2 idle checks
  if [ "$idle" -ge 2 ]; then echo "NO WORKERS on Amax"; exit 0; fi
  sleep 900
done
