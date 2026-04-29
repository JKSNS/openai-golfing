#!/usr/bin/env bash
# Phase 2: Anchor-Regularized TTT (ART) on top of CW-TTT.
# Adds an L2 penalty pulling theta toward theta_0 (deserialized weights at
# chunk 0). Prevents TTT drift from the calibrated quantized model.
#
# Args:  bash run_phase2.sh [SEED] [FLOOR] [ANCHOR_LAMBDA]
#        defaults: SEED=42, FLOOR=0.05, ANCHOR=1.0

set -euo pipefail

SEED="${1:-42}"
FLOOR="${2:-0.05}"
ANCHOR="${3:-1.0}"
WORKDIR="${WORKDIR:-/workspace}"
SUBMISSION_DIR="$WORKDIR/parameter-golf/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="cwart_seed${SEED}_f${FLOOR}_a${ANCHOR}_$(date +%Y%m%d_%H%M%S)"
LOG="$WORKDIR/logs/${RUN_ID}.log"
RUN_DIR="$HERE/runs/${RUN_ID}"

mkdir -p "$WORKDIR/logs" "$RUN_DIR"
echo "============================================================"
echo "[phase2] CW-TTT + Anchor-Regularized TTT (ART)"
echo "  seed   = $SEED   floor = $FLOOR   anchor_lambda = $ANCHOR"
echo "  log    = $LOG"
echo "  budget = ~18 min, ~\$7"
echo "============================================================"

cd "$SUBMISSION_DIR"
DATA_DIR="$WORKDIR/parameter-golf/data/" \
SEED="$SEED" \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
TTT_CONFIDENCE_WEIGHTED=1 TTT_WEIGHT_FLOOR="$FLOOR" \
TTT_ANCHOR_LAMBDA="$ANCHOR" \
RUN_ID="$RUN_ID" \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$LOG"

echo
python3 "$HERE/scripts/validate.py" "$LOG" || true
SEED="$SEED" QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  TTT_CONFIDENCE_WEIGHTED=1 TTT_WEIGHT_FLOOR="$FLOOR" TTT_ANCHOR_LAMBDA="$ANCHOR" \
  python3 "$HERE/scripts/report.py" "$LOG" --out "$RUN_DIR" >/dev/null

echo
echo "============================================================"
echo "[phase2] done"
echo "  log:    $LOG"
echo "  report: $RUN_DIR/report.md"
echo "============================================================"
