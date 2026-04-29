#!/usr/bin/env bash
# Live progress + cost + ETA for a running training log.
# Open a second terminal, `bash scripts/monitor.sh logs/repro_seed42.log`
#
# Tails the log, prints a single-line dashboard updating every 5s:
#   step 1234/4988 (24.7%) | 412s elapsed | ETA 873s | $2.33 | tok/s 95k

set -euo pipefail

LOG="${1:-}"
[ -z "$LOG" ] && LOG=$(ls -t /workspace/logs/*.log 2>/dev/null | head -1)
[ -z "${LOG:-}" ] || [ ! -f "$LOG" ] && { echo "usage: monitor.sh [logfile]"; exit 1; }

COST_PER_HR=25.0
clear
echo "[monitor] tailing $LOG (Ctrl-C to stop)"
echo

while :; do
  line=$(grep -E '^[0-9]+/[0-9]+ train_loss:' "$LOG" 2>/dev/null | tail -1)
  if [ -z "$line" ]; then
    printf "\r[monitor] waiting for first train step... %s" "$(date +%T)"
    sleep 3
    continue
  fi
  step=$(echo "$line" | awk -F'[/ ]' '{print $1}')
  total=$(echo "$line" | awk -F'[/ ]' '{print $2}')
  loss=$(echo "$line" | grep -oE 'train_loss:\s*[0-9.]+' | awk '{print $2}')
  tmin=$(echo "$line" | grep -oE 'train_time:\s*[0-9.]+m' | grep -oE '[0-9.]+')
  tps=$(echo "$line"  | grep -oE 'tok/s:\s*[0-9]+'    | grep -oE '[0-9]+$')
  pct=$(awk -v s="$step" -v t="$total" 'BEGIN{printf "%.1f", 100*s/t}')
  elapsed=$(awk -v m="$tmin" 'BEGIN{printf "%.0f", m*60}')
  eta=$(awk -v s="$step" -v t="$total" -v e="$elapsed" 'BEGIN{if(s>0)printf "%.0f", e*(t-s)/s; else print "-"}')
  cost=$(awk -v e="$elapsed" -v c="$COST_PER_HR" 'BEGIN{printf "%.2f", e/3600*c}')
  printf "\r[monitor] step %s/%s (%s%%) | loss %s | %ss elapsed | ETA %ss | \$%s | %s tok/s    " \
    "$step" "$total" "$pct" "$loss" "$elapsed" "$eta" "$cost" "$tps"
  sleep 5
done
