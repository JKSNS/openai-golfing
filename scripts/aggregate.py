#!/usr/bin/env python3
"""Aggregate quantized_ttt val_bpb across multiple training logs and report mean
+/- std plus the p<0.01 confidence margin needed for a record submission.

Usage:
  python3 scripts/aggregate.py logs/run_seed*.log
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path


VAL_BPB = re.compile(r"quantized_ttt[^\n]*val_bpb[\s:=]+(\d+\.\d+)")
SOTA = 1.0810   # PR #1493 current record
DELTA = 0.005   # required improvement per submission rules


def extract(path: Path) -> float | None:
    matches = VAL_BPB.findall(path.read_text(errors="replace"))
    return float(matches[-1]) if matches else None


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: aggregate.py LOG [LOG ...]")
        return 2

    rows: list[tuple[Path, float]] = []
    for arg in sys.argv[1:]:
        for p in sorted(Path().glob(arg)) or [Path(arg)]:
            v = extract(p)
            if v is None:
                print(f"  skip  {p}: no quantized_ttt val_bpb")
                continue
            rows.append((p, v))
            print(f"  ok    {p}: val_bpb={v}")

    if len(rows) < 2:
        print("need at least 2 valid logs")
        return 1

    vals = [v for _, v in rows]
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(var)
    sem = std / math.sqrt(n)

    # Two-sided p<0.01 critical t value, df=n-1
    t_crit = {2: 63.66, 3: 9.925, 4: 5.841, 5: 4.604, 10: 3.250}.get(n, 3.25)
    margin = t_crit * sem

    target = SOTA - DELTA
    upper_ci = mean + margin

    print()
    print(f"  n             = {n}")
    print(f"  mean          = {mean:.5f}")
    print(f"  std           = {std:.5f}")
    print(f"  sem           = {sem:.5f}")
    print(f"  p<0.01 margin = +/- {margin:.5f}")
    print(f"  upper-CI      = {upper_ci:.5f}")
    print(f"  SOTA target   = {target:.5f} ({SOTA} - {DELTA})")
    print()
    if upper_ci < target:
        print(f"  RECORD: upper-CI {upper_ci:.5f} beats target {target:.5f} at p<0.01")
        return 0
    if mean < target:
        print(f"  CLOSE: mean {mean:.5f} beats target but upper-CI {upper_ci:.5f} does not.")
        print(f"         Run more seeds to tighten the bound.")
        return 1
    print(f"  NOT RECORD: mean {mean:.5f} does not beat target {target:.5f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
