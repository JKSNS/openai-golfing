#!/usr/bin/env bash
# Phase 4: 3-seed validation. Runs the chosen config on seeds 42, 314, 999.
# Produces an aggregate report at runs/3seed_<config_tag>/aggregate.md
# with Welch t-test against PR #1493's published 1.0810 +/- 0.0007.
#
# Usage:
#   bash scripts/run_3seed.sh                        # cwttt floor=0.05
#   bash scripts/run_3seed.sh repro                  # bigbag baseline (no CW-TTT)
#   bash scripts/run_3seed.sh cwttt 0.10             # cwttt at floor=0.10
#   bash scripts/run_3seed.sh cwttt 0.05 "11 22 33"  # custom seed list
#
# Cost: ~$21-25 (3 runs at ~$7 each + small overhead).
# Time: ~60 min wallclock.

set -euo pipefail

MODE="${1:-cwttt}"     # repro | cwttt
FLOOR="${2:-0.05}"
SEEDS="${3:-42 314 999}"
WORKDIR="${WORKDIR:-/workspace}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

case "$MODE" in
  repro) tag="repro_${TS}" ;;
  cwttt) tag="cwttt_floor${FLOOR}_${TS}" ;;
  *) echo "[3seed] unknown mode '$MODE' (expected: repro | cwttt)"; exit 2 ;;
esac

OUT_DIR="$HERE/runs/3seed_${tag}"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "[3seed] $MODE  seeds=$SEEDS  floor=$FLOOR"
echo "  out_dir = $OUT_DIR"
echo "  cost    = ~\$21-25"
echo "  time    = ~60 min"
echo "============================================================"

logs=()
for SEED in $SEEDS; do
  echo
  echo "[3seed] -> seed $SEED"
  if [ "$MODE" = "repro" ]; then
    bash "$HERE/scripts/run_phase0.sh" "$SEED"
    log=$(ls -t "$WORKDIR"/logs/repro_seed${SEED}_*.log | head -1)
  else
    bash "$HERE/scripts/run_phase1.sh" "$SEED" "$FLOOR"
    log=$(ls -t "$WORKDIR"/logs/cwttt_seed${SEED}_floor${FLOOR}_*.log | head -1)
  fi
  cp "$log" "$OUT_DIR/seed${SEED}.log"
  logs+=("$OUT_DIR/seed${SEED}.log")
done

echo
echo "[3seed] aggregating $(echo "$SEEDS" | wc -w) seeds"
python3 "$HERE/scripts/aggregate.py" "${logs[@]}" | tee "$OUT_DIR/aggregate.txt"

# generate per-seed reports too
for log in "${logs[@]}"; do
  base=$(basename "$log" .log)
  python3 "$HERE/scripts/report.py" "$log" --out "$OUT_DIR/$base" >/dev/null
done

cat > "$OUT_DIR/aggregate.md" <<EOF
# 3-seed run: $MODE

- Mode: \`$MODE\`
- Floor: $FLOOR
- Seeds: $SEEDS
- Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- Per-seed reports: \`seed<N>/report.md\`

## Aggregate

\`\`\`
$(cat "$OUT_DIR/aggregate.txt")
\`\`\`
EOF

echo
echo "============================================================"
echo "[3seed] done"
echo "  aggregate: $OUT_DIR/aggregate.md"
echo "  reports  : $OUT_DIR/seed*/report.md"
echo "============================================================"
