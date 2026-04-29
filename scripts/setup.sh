#!/usr/bin/env bash
# One-time pod setup. Idempotent: rerun is a no-op once everything is in place.
# Installs pip deps, ensures FA3 is importable, downloads FineWeb SP8192,
# clones bigbag's fork, drops in the patched train_gpt.py.
#
# Fails loud at every prerequisite. No silent skip.

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
PG_REF="${PG_REF:-857de47}"
PG_FORK="${PG_FORK:-https://github.com/bigbag/parameter-golf.git}"
SP8192_HF_REPO="${SP8192_HF_REPO:-kevclark/parameter-golf}"
SUBMISSION_DIR="records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$WORKDIR" "$WORKDIR/logs"
cd "$WORKDIR"

step() { echo "[setup] $*"; }
fail() { echo "[setup] FAIL: $*" >&2; exit 1; }

# 1. clone bigbag fork (rm any stale or wrong-remote dir first)
step "step 1/6  cloning bigbag/parameter-golf @ $PG_REF"
if [ -d parameter-golf ]; then
  if [ ! -d parameter-golf/.git ]; then
    step "  removing stale parameter-golf (not a git repo)"
    rm -rf parameter-golf
  elif ! (cd parameter-golf && git remote get-url origin 2>/dev/null | grep -q "bigbag/parameter-golf"); then
    step "  removing parameter-golf (wrong remote)"
    rm -rf parameter-golf
  fi
fi
[ -d parameter-golf ] || git clone "$PG_FORK"
( cd parameter-golf && git fetch --all -q && git checkout "$PG_REF" -q )
step "  head: $(cd parameter-golf && git rev-parse --short HEAD)"

# 2. pip deps from bigbag requirements.txt
step "step 2/6  installing python deps"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r "$WORKDIR/parameter-golf/requirements.txt"
python3 -m pip install --quiet brotli

# 3. flash_attn_3 interface check + best-effort install
step "step 3/6  checking flash_attn_interface (FA3)"
if ! python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
  step "  not importable; attempting pip install flash-attn"
  python3 -m pip install --quiet flash-attn --no-build-isolation || true
  if ! python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    cat <<EOF >&2

[setup] FA3 (flash_attn_interface) not available on this pod. Bigbag's submission
imports it at the top of train_gpt.py and cannot run without it.

You have two options:

  (a) Use the official Parameter Golf RunPod template instead of a generic 8xH100.
      That template has FA3 + Python 3.12 + all deps pre-installed.

  (b) Compile FA3 from source on this pod (takes ~30 minutes, needs CUDA toolkit):
        cd /tmp
        git clone https://github.com/Dao-AILab/flash-attention
        cd flash-attention/hopper && python setup.py install
      Then rerun: bash $HERE/scripts/setup.sh

(a) is faster and what bigbag and everyone else on the leaderboard uses.
EOF
    exit 1
  fi
fi
step "  flash_attn_interface: ok"

# 4. CUDA + Hopper
step "step 4/6  verifying CUDA + Hopper"
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
n = torch.cuda.device_count()
print(f"  CUDA devices: {n}")
hopper_count = 0
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name} cc={p.major}.{p.minor}")
    if p.major == 9: hopper_count += 1
if hopper_count == 0:
    raise SystemExit("  no Hopper GPUs detected (need cc 9.x for FA3)")
PY

# 5. SP8192 dataset (from kevclark, not willdepueoai default)
step "step 5/6  prefetching FineWeb SP8192 from $SP8192_HF_REPO"
DATASET_DIR="$WORKDIR/parameter-golf/data/datasets/fineweb10B_sp8192"
TOKENIZER_PATH="$WORKDIR/parameter-golf/data/tokenizers/fineweb_8192_bpe.model"
if [ ! -f "$TOKENIZER_PATH" ] || [ ! -f "$DATASET_DIR/fineweb_val_000000.bin" ]; then
  cd "$WORKDIR/parameter-golf"
  MATCHED_FINEWEB_REPO_ID="$SP8192_HF_REPO" \
    python3 data/cached_challenge_fineweb.py --variant sp8192
  cd "$WORKDIR"
fi
val_count=$(ls "$DATASET_DIR"/fineweb_val_*.bin 2>/dev/null | wc -l)
train_count=$(ls "$DATASET_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
[ "$val_count" -ge 1 ] || fail "no SP8192 val shards downloaded to $DATASET_DIR"
[ "$train_count" -ge 1 ] || fail "no SP8192 train shards downloaded to $DATASET_DIR"
[ -f "$TOKENIZER_PATH" ] || fail "tokenizer missing at $TOKENIZER_PATH"
step "  val=$val_count train=$train_count tokenizer=ok"

# 6. drop patched train_gpt.py
step "step 6/6  installing patched train_gpt.py over bigbag's broken-on-3.11 version"
cp "$HERE/patches/train_gpt.py" \
   "$WORKDIR/parameter-golf/$SUBMISSION_DIR/train_gpt.py"
step "  TTT_CONFIDENCE_WEIGHTED=0 (default) -> bigbag baseline"
step "  TTT_CONFIDENCE_WEIGHTED=1           -> CW-TTT angle"

step "done"
