#!/bin/bash
# Quick ablation: run two configs and compare
# Usage: bash scripts/quick_ablation.sh
# Designed for 1xH100 — runs ~2-3 minutes each

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

DATA_PATH=./upstream/data/datasets/fineweb10B_sp1024/
TOKENIZER_PATH=./upstream/data/tokenizers/fineweb_1024_bpe.model

echo "=== Quick Ablation Test ==="
echo "Running 500 steps each on 1 GPU"
echo ""

# Baseline config
echo "--- Config A: Baseline ---"
RUN_ID=ablation_baseline \
ITERATIONS=500 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=100 \
DATA_PATH=$DATA_PATH \
TOKENIZER_PATH=$TOKENIZER_PATH \
VOCAB_SIZE=1024 \
SLOT_STEPS=0 \
torchrun --standalone --nproc_per_node=1 src/train_gpt.py 2>&1 | tail -20

echo ""
echo "--- Config B: With modifications (edit this section) ---"
# Modify env vars below to test your changes
RUN_ID=ablation_modified \
ITERATIONS=500 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=100 \
DATA_PATH=$DATA_PATH \
TOKENIZER_PATH=$TOKENIZER_PATH \
VOCAB_SIZE=1024 \
SLOT_STEPS=0 \
torchrun --standalone --nproc_per_node=1 src/train_gpt.py 2>&1 | tail -20

echo ""
echo "=== Compare the final val_bpb values above ==="
