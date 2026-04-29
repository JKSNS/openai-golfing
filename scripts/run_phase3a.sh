#!/usr/bin/env bash
# Phase 3a: floor sweep on the CW-TTT angle.
# 4 runs (floors 0.0, 0.05, 0.10, 0.25), single seed. ~$28 total.
# Only run after Phase 1 shows >=0.001 nat improvement on seed 42.

set -euo pipefail

SEED="${1:-42}"
WORKDIR="${WORKDIR:-/workspace}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for FLOOR in 0.0 0.05 0.10 0.25; do
  echo "==== floor=$FLOOR ===="
  bash "$HERE/scripts/run_phase1.sh" "$SEED" "$FLOOR"
done

python3 "$HERE/scripts/compare.py" "$WORKDIR/logs/cwttt_seed${SEED}"_floor*.log
echo
echo "Pick the best floor from the table above. Then 3-seed validate with:"
echo "  for S in 42 314 999; do bash $HERE/scripts/run_phase1.sh \$S <best_floor>; done"
echo "  python3 $HERE/scripts/aggregate.py $WORKDIR/logs/cwttt_seed*_floor<best>.log"
