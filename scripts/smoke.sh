#!/usr/bin/env bash
# Smoke test: 60 training steps + skip eval. ~30-60 seconds. ~$0.20.
# Validates the full pipeline (data load, forward, backward, optimizer,
# distributed comms) without burning a real run. Use after preflight to
# catch FA3-specific failures or NCCL issues before the real Phase 0/1.
#
#   bash scripts/smoke.sh

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
SUBMISSION_DIR="$WORKDIR/parameter-golf/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$WORKDIR/logs/smoke.log"
mkdir -p "$WORKDIR/logs"

echo "[smoke] 60-step micro-train, no eval. ~30-60s. log=$LOG"

cd "$SUBMISSION_DIR"
DATA_DIR="$WORKDIR/parameter-golf/data/" \
SEED=42 \
ITERATIONS=60 \
MAX_WALLCLOCK_SECONDS=120 \
WARMUP_STEPS=0 \
TRAIN_LOG_EVERY=10 \
VAL_LOSS_EVERY=0 \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$LOG"

# crude pass check
if grep -qE 'train_loss:.*[0-9]' "$LOG"; then
  echo "[smoke] PASS: training advanced past step 1"
  exit 0
fi
echo "[smoke] FAIL: no train_loss line in log; check $LOG"
exit 1
