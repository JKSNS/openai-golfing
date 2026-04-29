#!/usr/bin/env bash
# Phase 2b: full Stabilized TTT stack: CW + ART + EMA-Teacher.
# Adds KL(student || EMA_teacher) auxiliary loss on top of Phase 2.
# EMA tracks the trajectory of the TTT model and acts as a stability signal.
#
# Args:  bash run_phase2b.sh [SEED] [FLOOR] [ANCHOR] [EMA_TEACHER] [EMA_DECAY]
#        defaults: SEED=42, FLOOR=0.05, ANCHOR=1.0, EMA=0.5, DECAY=0.99

set -euo pipefail

SEED="${1:-42}"
FLOOR="${2:-0.05}"
ANCHOR="${3:-1.0}"
EMA_LAMBDA="${4:-0.5}"
EMA_DECAY="${5:-0.99}"
WORKDIR="${WORKDIR:-/workspace}"
SUBMISSION_DIR="$WORKDIR/parameter-golf/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="stabttt_seed${SEED}_f${FLOOR}_a${ANCHOR}_e${EMA_LAMBDA}_$(date +%Y%m%d_%H%M%S)"
LOG="$WORKDIR/logs/${RUN_ID}.log"
RUN_DIR="$HERE/runs/${RUN_ID}"

mkdir -p "$WORKDIR/logs" "$RUN_DIR"
echo "============================================================"
echo "[phase2b] Full Stabilized TTT stack: CW + ART + EMA-Teacher"
echo "  seed = $SEED  floor = $FLOOR  anchor = $ANCHOR"
echo "  ema_lambda = $EMA_LAMBDA  ema_decay = $EMA_DECAY"
echo "  budget = ~20 min, ~\$8"
echo "============================================================"

cd "$SUBMISSION_DIR"
DATA_DIR="$WORKDIR/parameter-golf/data/" \
SEED="$SEED" \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
TTT_CONFIDENCE_WEIGHTED=1 TTT_WEIGHT_FLOOR="$FLOOR" \
TTT_ANCHOR_LAMBDA="$ANCHOR" \
TTT_EMA_TEACHER_LAMBDA="$EMA_LAMBDA" TTT_EMA_DECAY="$EMA_DECAY" \
RUN_ID="$RUN_ID" \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$LOG"

echo
python3 "$HERE/scripts/validate.py" "$LOG" || true
SEED="$SEED" QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  TTT_CONFIDENCE_WEIGHTED=1 TTT_WEIGHT_FLOOR="$FLOOR" TTT_ANCHOR_LAMBDA="$ANCHOR" \
  TTT_EMA_TEACHER_LAMBDA="$EMA_LAMBDA" TTT_EMA_DECAY="$EMA_DECAY" \
  python3 "$HERE/scripts/report.py" "$LOG" --out "$RUN_DIR" >/dev/null

echo
echo "============================================================"
echo "[phase2b] done"
echo "  log:    $LOG"
echo "  report: $RUN_DIR/report.md"
echo "============================================================"
