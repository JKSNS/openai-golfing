#!/bin/bash
# Run 3 seeds for statistical validation (required for leaderboard submission)
# Usage: bash scripts/three_seed_validation.sh
# Cost: ~$10.50 (3 × $3.50)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== 3-Seed Validation (8xH100) ==="
echo "This will run 3 full training+eval cycles"
echo "Total time: ~60 minutes, Cost: ~$10.50"
echo ""

for SEED in 1337 42 314; do
    echo "========================================="
    echo "=== Seed $SEED ==="
    echo "========================================="

    RUN_ID="validation_seed${SEED}" \
    SEED=$SEED \
    DATA_PATH=./upstream/data/datasets/fineweb10B_sp1024/ \
    TOKENIZER_PATH=./upstream/data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 \
    SLOT_STEPS=16 \
    SLOT_LR=0.008 \
    SLOT_LR_MIN=0.0008 \
    torchrun --standalone --nproc_per_node=8 src/train_gpt.py 2>&1 | tee "experiments/validation_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
done

echo "=== All 3 seeds complete ==="
echo "Grep for 'val_bpb' in experiments/validation_seed*.log to compare"
echo ""
grep -h "val_bpb" experiments/validation_seed*.log 2>/dev/null | tail -6
