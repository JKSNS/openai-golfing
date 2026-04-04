#!/bin/bash
# Smoke test: verify the training script runs correctly
# Run this on ANY machine (even CPU, though it'll be slow)
# Usage: bash scripts/smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== Parameter Golf Smoke Test ==="
echo "Working dir: $SCRIPT_DIR"

# Check if data exists
if [ ! -d "upstream/data/datasets/fineweb10B_sp1024" ]; then
    echo "Data not found. Downloading 1 shard for testing..."
    python3 upstream/data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
fi

# Quick training run: 50 iterations, no periodic validation
echo ""
echo "=== Starting smoke test (50 iterations) ==="
RUN_ID=smoke_test \
ITERATIONS=50 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=10 \
TRAIN_BATCH_TOKENS=8192 \
DATA_PATH=./upstream/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./upstream/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SLOT_STEPS=0 \
torchrun --standalone --nproc_per_node=1 src/train_gpt.py

echo ""
echo "=== Smoke test complete ==="
echo "Check the output above for:"
echo "  - Loss decreasing over steps"
echo "  - Final val_bpb printed"
echo "  - Artifact size under 16MB"
