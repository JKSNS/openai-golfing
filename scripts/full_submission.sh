#!/bin/bash
# Full submission run on 8xH100
# Usage: bash scripts/full_submission.sh [seed]
# Cost: ~$3.50 per run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

SEED=${1:-1337}

echo "=== Full Submission Run (8xH100) ==="
echo "Seed: $SEED"
echo "This will take ~10 minutes training + ~10 minutes eval"
echo ""

RUN_ID="submission_seed${SEED}" \
SEED=$SEED \
DATA_PATH=./upstream/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./upstream/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SLOT_LBFGS=1 \
SLOT_LBFGS_MAX_ITER=10 \
SLOT_WARMSTART_ALPHA=0.85 \
SLOT_DELTA_CLIP=5.0 \
SLOT_FOCAL_TOKENS=128 \
TTT_ENABLED=1 \
TTT_EPOCHS=6 \
TTT_FREEZE_BLOCKS=2 \
RECUR_ENABLED=1 \
RECUR_LAYERS=4,5 \
RECUR_START_STEP=3000 \
MUON_EQ_R=1 \
MUON_WD=0.09 \
QK_GAIN_INIT=5.0 \
torchrun --standalone --nproc_per_node=8 src/train_gpt.py 2>&1 | tee "experiments/run_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=== Run complete. Check experiments/ for the log ==="
