#!/usr/bin/env bash
# Phase 0: baseline reproduction of bigbag's PR #1493 (val_bpb ~= 1.0810).
# Single seed. ~16 min on 8xH100 SXM. ~$7.
#
# Uses the patched train_gpt.py (CW-TTT off by default = byte-equivalent
# runtime behavior to bigbag's original).
#
# Args:  bash run_phase0.sh [SEED]
#        defaults: SEED=42

set -euo pipefail

SEED="${1:-42}"
WORKDIR="${WORKDIR:-/workspace}"
SUBMISSION_DIR="$WORKDIR/parameter-golf/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="repro_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
LOG="$WORKDIR/logs/${RUN_ID}.log"
RUN_DIR="$HERE/runs/${RUN_ID}"

mkdir -p "$WORKDIR/logs" "$RUN_DIR"
echo "============================================================"
echo "[phase0] reproduce PR #1493 baseline"
echo "  seed     = $SEED"
echo "  log      = $LOG"
echo "  run_dir  = $RUN_DIR"
echo "  expected = quantized_ttt val_bpb ~= 1.0810"
echo "  budget   = ~16 min, ~\$7"
echo "  monitor  = bash $HERE/scripts/monitor.sh $LOG  (in another terminal)"
echo "============================================================"

start_ts=$(date +%s)

cd "$SUBMISSION_DIR"
DATA_DIR="$WORKDIR/parameter-golf/data/" \
SEED="$SEED" \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
TTT_CONFIDENCE_WEIGHTED=0 \
RUN_ID="$RUN_ID" \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$LOG"

end_ts=$(date +%s)
elapsed=$((end_ts - start_ts))

echo
echo "[phase0] gates"
python3 "$HERE/scripts/validate.py" "$LOG" || true

echo
echo "[phase0] generating report at $RUN_DIR/"
SEED="$SEED" QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 TTT_CONFIDENCE_WEIGHTED=0 \
  python3 "$HERE/scripts/report.py" "$LOG" --out "$RUN_DIR" >/dev/null

echo
echo "============================================================"
echo "[phase0] done  ($((elapsed/60))m $((elapsed%60))s wall)"
echo "  log   : $LOG"
echo "  report: $RUN_DIR/report.md"
echo "  json  : $RUN_DIR/report.json"
echo "============================================================"
