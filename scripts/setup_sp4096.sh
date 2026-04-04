#!/bin/bash
# Download SP4096 tokenizer and data for larger-vocabulary experiments
# Usage: bash scripts/setup_sp4096.sh [train-shards]
# Default: 10 shards (~1B tokens). Use 80 for full dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

SHARDS=${1:-10}

echo "=== Downloading SP4096 FineWeb data ==="
echo "Train shards: $SHARDS"
echo ""

python3 upstream/data/cached_challenge_fineweb.py --variant sp4096 --train-shards "$SHARDS"

echo ""
echo "=== SP4096 data ready ==="
echo ""
echo "To train with SP4096, use:"
echo "  VOCAB_SIZE=4096 \\"
echo "  DATA_PATH=./upstream/data/datasets/fineweb10B_sp4096/ \\"
echo "  TOKENIZER_PATH=./upstream/data/tokenizers/fineweb_4096_bpe.model \\"
echo "  MLP_MULT=4.0 \\"
echo "  torchrun --standalone --nproc_per_node=8 src/train_gpt.py"
