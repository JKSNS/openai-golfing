#!/usr/bin/env bash
# Pre-flight check. Validates the pod environment without burning compute.
# Run before any phase. Catches FA3 missing, data missing, NVLink broken,
# disk full, etc. before you spend $7 finding out.
#
#   bash scripts/preflight.sh
#
# Exit 0 if all green. Exit 1 if anything is wrong.

set -uo pipefail

WORKDIR="${WORKDIR:-/workspace}"
SUBMISSION_DIR="$WORKDIR/parameter-golf/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT"
DATASET_DIR="$WORKDIR/parameter-golf/data/datasets/fineweb10B_sp8192"
TOKENIZER="$WORKDIR/parameter-golf/data/tokenizers/fineweb_8192_bpe.model"

ok=0; warn=0; fail=0
g() { echo "  ok    $*"; ok=$((ok+1)); }
w() { echo "  WARN  $*"; warn=$((warn+1)); }
f() { echo "  FAIL  $*"; fail=$((fail+1)); }

echo "[preflight] python"
python3 -c "import sys; assert sys.version_info >= (3,12), sys.version" 2>/dev/null \
  && g "python >= 3.12 ($(python3 -V 2>&1 | awk '{print $2}'))" \
  || w "python < 3.12 ($(python3 -V 2>&1 | awk '{print $2}')); bigbag's f-strings need 3.12+"

echo "[preflight] gpu"
n_gpus=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
[ "$n_gpus" = "8" ] && g "8 GPUs visible" || f "$n_gpus GPUs visible (expected 8)"

python3 - 2>/dev/null <<'PY' && g "all GPUs are Hopper (cc 9.x)" || f "non-Hopper GPU detected"
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    assert p.major == 9, f"GPU{i} is cc{p.major}.{p.minor} not 9.x"
PY

echo "[preflight] flash_attn_3"
python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null \
  && g "flash_attn_interface importable" \
  || f "flash_attn_interface missing (install flash-attn or use the official RunPod template)"

echo "[preflight] nvlink topology"
nv_pairs=$(nvidia-smi topo -m 2>/dev/null | awk 'NR>1 && $1 ~ /^GPU/ {for (i=2;i<=9;i++) print $i}' | grep -c '^NV' || true)
if [ "$nv_pairs" -ge 50 ]; then
  g "NVLink mesh ($nv_pairs NV-pair entries; expected ~56 for full 8x mesh)"
else
  w "weak NVLink ($nv_pairs NV-pair entries); set NCCL_P2P_DISABLE=1 if runs crash"
fi

echo "[preflight] data"
[ -f "$TOKENIZER" ] && g "tokenizer at $TOKENIZER" || f "tokenizer missing: $TOKENIZER"
val_count=$(ls "$DATASET_DIR"/fineweb_val_*.bin 2>/dev/null | wc -l)
train_count=$(ls "$DATASET_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
[ "$val_count" -ge 1 ]   && g "val shards: $val_count"   || f "val shards: $val_count (expected >=1)"
[ "$train_count" -ge 80 ] && g "train shards: $train_count" \
  || w "train shards: $train_count (expected 80; small ablations OK with fewer)"

echo "[preflight] disk"
free_gb=$(df -BG "$WORKDIR" | awk 'NR==2 {print $4}' | tr -d 'G')
[ "${free_gb:-0}" -ge 30 ] && g "free disk: ${free_gb}G" || w "free disk: ${free_gb}G (recommend >=30G for logs/checkpoints)"

echo "[preflight] gpu memory"
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
[ "${free_mib:-0}" -ge 70000 ] && g "GPU0 free memory: ${free_mib} MiB" \
  || w "GPU0 free memory: ${free_mib} MiB; another process may be holding GPU"

echo "[preflight] submission dir"
[ -f "$SUBMISSION_DIR/train_gpt.py" ] && g "patched train_gpt.py in place" \
  || f "train_gpt.py missing at $SUBMISSION_DIR; rerun setup.sh"

echo
echo "[preflight] summary: ok=$ok warn=$warn fail=$fail"
[ "$fail" -eq 0 ] && exit 0 || exit 1
