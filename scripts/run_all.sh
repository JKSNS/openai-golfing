#!/usr/bin/env bash
# All-in-one driver for the full Stabilized TTT ablation series.
#
# Sequence:
#   1. setup.sh            (~5 min, idempotent)
#   2. preflight.sh        (~5 sec)
#   3. run_phase0.sh       (~16 min, ~$7)  baseline reproduction
#   4. run_phase1.sh       (~18 min, ~$7)  + CW-TTT
#   5. run_phase2.sh       (~18 min, ~$7)  + CW + ART
#   6. run_phase2b.sh      (~20 min, ~$8)  + CW + ART + EMA-Teacher
#   7. compare.py          (table + winner)
#
# Total: ~80 min wallclock, ~$29 compute.
#
# Optional 3-seed validation on the winner:
#   RUN_3SEED=1 bash scripts/run_all.sh    (+$24, +60 min)
#
# Skip flags (idempotent reruns):
#   SKIP_SETUP=1 SKIP_PREFLIGHT=1 SKIP_P0=1 SKIP_P1=1 SKIP_P2=1 SKIP_P2B=1
#
# Recommended: open a second terminal during long phases:
#   bash scripts/monitor.sh /workspace/logs/<latest>.log

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-/workspace}"
SEED="${SEED:-42}"
FLOOR="${FLOOR:-0.05}"
ANCHOR="${ANCHOR:-1.0}"
EMA_LAMBDA="${EMA_LAMBDA:-0.5}"
EMA_DECAY="${EMA_DECAY:-0.99}"

step() {
  echo
  echo "============================================================"
  echo "[run_all] $*"
  echo "============================================================"
}

start_ts=$(date +%s)

if [ "${SKIP_SETUP:-0}" != "1" ]; then
  step "1/7 Setup"
  bash "$HERE/scripts/setup.sh"
fi

if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  step "2/7 Preflight"
  bash "$HERE/scripts/preflight.sh"
fi

if [ "${SKIP_P0:-0}" != "1" ]; then
  step "3/7 Phase 0: bigbag PR #1493 baseline reproduction (~16 min, ~\$7)"
  bash "$HERE/scripts/run_phase0.sh" "$SEED"
fi

if [ "${SKIP_P1:-0}" != "1" ]; then
  step "4/7 Phase 1: + CW-TTT (~18 min, ~\$7)"
  bash "$HERE/scripts/run_phase1.sh" "$SEED" "$FLOOR"
fi

if [ "${SKIP_P2:-0}" != "1" ]; then
  step "5/7 Phase 2: + CW + ART (~18 min, ~\$7)"
  bash "$HERE/scripts/run_phase2.sh" "$SEED" "$FLOOR" "$ANCHOR"
fi

if [ "${SKIP_P2B:-0}" != "1" ]; then
  step "6/7 Phase 2b: + CW + ART + EMA-Teacher (full Stabilized TTT, ~20 min, ~\$8)"
  bash "$HERE/scripts/run_phase2b.sh" "$SEED" "$FLOOR" "$ANCHOR" "$EMA_LAMBDA" "$EMA_DECAY"
fi

step "7/7 Compare"
python3 "$HERE/scripts/compare.py" \
  "$WORKDIR"/logs/repro_seed${SEED}_*.log \
  "$WORKDIR"/logs/cwttt_seed${SEED}_*.log \
  "$WORKDIR"/logs/cwart_seed${SEED}_*.log \
  "$WORKDIR"/logs/stabttt_seed${SEED}_*.log

end_ts=$(date +%s)
elapsed=$((end_ts - start_ts))

# decide winner: lowest val_bpb that beats baseline by >= 0.001
echo
echo "============================================================"
echo "[run_all] decision"
echo "============================================================"
python3 - <<PY
import re, glob, sys
SOTA = 1.0810
val_bpb = re.compile(r"quantized_ttt[^\n]*val_bpb[\s:=]+(\d+\.\d+)")
def last(p):
    matches = val_bpb.findall(open(p, errors='replace').read())
    return float(matches[-1]) if matches else None

phases = {
    "Phase 0 baseline": glob.glob("$WORKDIR/logs/repro_seed${SEED}_*.log"),
    "Phase 1 CW":       glob.glob("$WORKDIR/logs/cwttt_seed${SEED}_*.log"),
    "Phase 2 CW+ART":   glob.glob("$WORKDIR/logs/cwart_seed${SEED}_*.log"),
    "Phase 2b CW+ART+EMA": glob.glob("$WORKDIR/logs/stabttt_seed${SEED}_*.log"),
}
results = {}
for name, files in phases.items():
    files = sorted(files)
    if not files: continue
    v = last(files[-1])
    if v is not None: results[name] = v

if not results:
    print("  no results parsed")
    sys.exit(0)

baseline = results.get("Phase 0 baseline")
print(f"  {'phase':<28s} {'val_bpb':>9s}  {'vs base':>9s}  {'vs SOTA':>9s}")
for name, v in results.items():
    delta_b = (v - baseline) if baseline else None
    delta_s = v - SOTA
    db = f"{delta_b:+.5f}" if delta_b is not None else "-"
    print(f"  {name:<28s} {v:>9.5f}  {db:>9s}  {delta_s:+.5f}")

# pick winner: lowest val_bpb, must beat baseline by >=0.001 to count
winner = None
winner_v = float('inf')
for name, v in results.items():
    if name == "Phase 0 baseline": continue
    if baseline and v < baseline - 0.001 and v < winner_v:
        winner = name; winner_v = v

print()
if winner:
    print(f"  WINNER: {winner} at {winner_v:.5f} (delta_baseline {winner_v - baseline:+.5f})")
    print(f"  Next: 3-seed validate. bash scripts/run_3seed.sh <mode> <args>")
elif baseline and any(v < baseline - 0.0005 for v in results.values()):
    print("  CLOSE: some delta < -0.0005 but not >= -0.001. Consider another seed.")
else:
    print("  NONE: no novel variant clearly beats baseline. Submit Phase 0 reproduction as non-record.")
PY

echo
echo "============================================================"
echo "[run_all] complete  ($((elapsed/60))m $((elapsed%60))s wall)"
echo "  logs   : $WORKDIR/logs/"
echo "  reports: $HERE/runs/"
echo "============================================================"

# optional: 3-seed validate winner
if [ "${RUN_3SEED:-0}" = "1" ]; then
  step "Bonus: 3-seed validation of winning config"
  echo "[run_all] not implemented in this single-script flow."
  echo "[run_all] inspect the table above, then manually:"
  echo "  bash scripts/run_3seed.sh cwttt $FLOOR     # if Phase 1 won"
  echo "  bash scripts/run_3seed.sh repro            # if nothing beat baseline"
fi
