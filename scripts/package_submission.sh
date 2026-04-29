#!/usr/bin/env bash
# Build a records/ folder ready for PR to openai/parameter-golf.
# Takes a 3-seed run dir (from run_3seed.sh) and assembles:
#   submission/<date>_<name>/
#     train_gpt.py            (the encoded submission)
#     README.md               (compliance walkthrough + results)
#     submission.json         (GitHub handle, val_bpb, std, seed list)
#     train_seed<N>.log       (one per seed)
#
# Usage:
#   bash scripts/package_submission.sh runs/3seed_cwttt_floor0.05_<ts>
#   GITHUB_HANDLE=jksns GITHUB_NAME="Jackson Stephens" \
#       bash scripts/package_submission.sh runs/3seed_cwttt_floor0.05_<ts>

set -euo pipefail

SRC_DIR="${1:?usage: package_submission.sh <runs/3seed_*> [SUBMISSION_NAME]}"
NAME="${2:-2026-04-29_SP8192_Confidence_Weighted_TTT}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$HERE/submission/${NAME}"
GITHUB_HANDLE="${GITHUB_HANDLE:-jksns}"
GITHUB_NAME="${GITHUB_NAME:-Jackson Stephens}"

[ -d "$SRC_DIR" ] || { echo "FAIL: $SRC_DIR not found"; exit 1; }

# pull val_bpb numbers from the per-seed reports
mapfile -t bpbs < <(python3 - "$SRC_DIR" <<'PY'
import json, sys
from pathlib import Path
src = Path(sys.argv[1])
for d in sorted(src.glob("seed*")):
    j = d / "report.json"
    if j.is_file():
        data = json.loads(j.read_text())
        v = data.get("final_val_bpb")
        if v is not None:
            print(v)
PY
)

if [ "${#bpbs[@]}" -lt 1 ]; then
  echo "FAIL: no val_bpb numbers in $SRC_DIR/seed*/report.json"
  exit 1
fi

mean=$(python3 -c "vs=[$(IFS=','; echo "${bpbs[*]}")]; print(f'{sum(vs)/len(vs):.5f}')")
std=$(python3  -c "import math; vs=[$(IFS=','; echo "${bpbs[*]}")]; m=sum(vs)/len(vs); v=sum((x-m)**2 for x in vs)/(max(len(vs)-1,1)); print(f'{math.sqrt(v):.5f}')")
n=${#bpbs[@]}

mkdir -p "$OUT_DIR"
cp "$HERE/patches/train_gpt.py" "$OUT_DIR/train_gpt.py"
i=0
for log in "$SRC_DIR"/seed*.log; do
  cp "$log" "$OUT_DIR/$(basename "$log")"
  i=$((i+1))
done

cat > "$OUT_DIR/submission.json" <<JSON
{
  "name": "$NAME",
  "author_github": "$GITHUB_HANDLE",
  "author_name": "$GITHUB_NAME",
  "val_bpb_mean": $mean,
  "val_bpb_std": $std,
  "n_seeds": $n,
  "seeds": [$(IFS=','; echo "${bpbs[*]}")],
  "base_pr": 1493,
  "base_author": "bigbag",
  "tagline": "SP8192 + 3L recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT + Confidence-Weighted TTT (novel)"
}
JSON

cat > "$OUT_DIR/README.md" <<EOF
# Confidence-Weighted Score-First TTT

3-seed val_bpb: **$mean +/- $std** ($n seeds).

Built on top of bigbag's PR #1493 (SOTA at 1.0810). The only diff is in the
score-first TTT train phase: replace uniform CE with \`(1 - p_correct)\`-weighted
CE so the optimizer focuses on tokens the model is uncertain about. Per-token
weights are computed from the model's predictions on the chunk it is already
training on; no new validation data is touched.

## Compliance

The four PR #1413 reviewer conditions:

1. **Causality.** Per-token weights are computed under \`torch.no_grad()\`
   in the train phase, on the same chunk the train phase is already legally
   training on (PR #461 score-first pattern). No future tokens leak back.
2. **Normalized softmax.** Unchanged.
3. **Score-before-update.** The \`loss_sum\` accumulator that drives val_bpb
   is populated entirely in the score phase, before any weight is computed
   and before any train step runs.
4. **Single-pass.** Each chunk is scored exactly once. Per-token weights are
   passed downstream into the train phase without altering the score.

## Diff against PR #1493

Two new env vars (default-off):

- \`TTT_CONFIDENCE_WEIGHTED\` (0|1, default 0)
- \`TTT_WEIGHT_FLOOR\` (float, default 0.05)

When \`TTT_CONFIDENCE_WEIGHTED=0\` the runtime path is byte-equivalent to
PR #1493. When \`TTT_CONFIDENCE_WEIGHTED=1\`, the train inner loop becomes:

\`\`\`python
with torch.no_grad():
    nll_eval = F.cross_entropy(base_model.forward_logits(x), y, reduction='none')
    weights  = (1.0 - torch.exp(-nll_eval)).clamp(min=h.ttt_weight_floor)
nll_train = F.cross_entropy(base_model.forward_logits(x), y, reduction='none')
loss = (nll_train * weights).mean()
\`\`\`

instead of the original:

\`\`\`python
loss = base_model(x, y)   # mean-reduced CE
\`\`\`

Cost: one extra inference-mode forward per train step. Eval wall is ~90s
longer than baseline. Still well under the 600s eval budget.

## Reproduce

\`\`\`bash
git clone https://github.com/JKSNS/openai-golfing
cd openai-golfing
bash scripts/setup.sh
SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \\
  TTT_CONFIDENCE_WEIGHTED=1 TTT_WEIGHT_FLOOR=0.05 \\
  torchrun --standalone --nproc_per_node=8 train_gpt.py
\`\`\`

## Per-seed scores

$(for v in "${bpbs[@]}"; do echo "- $v"; done)

## Logs

One log per seed in this folder: \`train_seed<N>.log\`.
EOF

echo
echo "[package] wrote $OUT_DIR/"
ls -la "$OUT_DIR"
echo
echo "Open a PR with:"
echo "  cd $OUT_DIR/.. && gh pr create --repo openai/parameter-golf \\"
echo "    --title \"$NAME\" --body-file $OUT_DIR/README.md"
