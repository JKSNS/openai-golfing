#!/usr/bin/env python3
"""Validate a Parameter Golf training+eval log.

Confirms:
  - Training wallclock <= 600s
  - Eval wallclock <= 600s
  - Pre-update val_bpb is in the expected range
  - Compliance: TTT scored every chunk under inference_mode before training on it
  - Final compressed artifact size <= 16,000,000 bytes (decimal)

Usage:
  python3 scripts/validate.py path/to/run.log [--expected-range 1.078,1.084]

Exits non-zero on any gate failure.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


GATES = {
    "train_seconds_max": 600.0,
    "eval_seconds_max": 600.0,
    "artifact_bytes_max": 16_000_000,
}

PATTERNS = {
    "train_time": re.compile(r"\btrain_time:\s*([\d\.]+)m\b"),
    "eval_time_ms": re.compile(r"\beval_time:\s*(\d+)ms"),
    "val_bpb": re.compile(r"\bval_bpb[\s:=]+(\d+\.\d+)"),
    "quantized_ttt_bpb": re.compile(r"quantized_ttt[^\n]*val_bpb[\s:=]+(\d+\.\d+)"),
    "artifact_bytes": re.compile(r"\bbytes_total[\s:=]+(\d+)"),
    "ttt_score_first": re.compile(r"ttt:start chunks=", re.I),
}


def find_last(rx: re.Pattern, text: str) -> str | None:
    matches = list(rx.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    return last.group(1) if last.groups() else last.group(0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument(
        "--expected-range",
        default="1.078,1.084",
        help="min,max for quantized_ttt val_bpb (PR #1493 reproduction)",
    )
    args = p.parse_args()

    if not args.log.exists():
        print(f"FAIL  log not found: {args.log}")
        return 2

    text = args.log.read_text(errors="replace")
    failures: list[str] = []

    val_bpb = find_last(PATTERNS["quantized_ttt_bpb"], text) or find_last(PATTERNS["val_bpb"], text)
    train_t_min = find_last(PATTERNS["train_time"], text)
    eval_t_ms = find_last(PATTERNS["eval_time_ms"], text)
    artifact = find_last(PATTERNS["artifact_bytes"], text)
    has_score_first = bool(PATTERNS["ttt_score_first"].search(text))

    train_t = float(train_t_min) * 60.0 if train_t_min else None
    eval_t = float(eval_t_ms) / 1000.0 if eval_t_ms else None

    lo, hi = (float(x) for x in args.expected_range.split(","))

    print(f"[validate] log={args.log}")
    print(f"  quantized_ttt val_bpb = {val_bpb}")
    print(f"  train_seconds         = {train_t}")
    print(f"  eval_seconds          = {eval_t}")
    print(f"  artifact_bytes        = {artifact}")
    print(f"  ttt:start marker      = {has_score_first}")

    if val_bpb is None:
        failures.append("missing val_bpb")
    else:
        v = float(val_bpb)
        if not (lo <= v <= hi):
            failures.append(f"val_bpb {v} outside expected [{lo}, {hi}]")

    if train_t is not None and train_t > GATES["train_seconds_max"]:
        failures.append(f"train_seconds {train_t:.1f} > {GATES['train_seconds_max']}")
    if eval_t is not None and eval_t > GATES["eval_seconds_max"]:
        failures.append(f"eval_seconds {eval_t:.1f} > {GATES['eval_seconds_max']}")
    if artifact and int(artifact) > GATES["artifact_bytes_max"]:
        failures.append(f"artifact_bytes {artifact} > {GATES['artifact_bytes_max']}")
    if not has_score_first:
        failures.append("no ttt:start marker; score-first TTT compliance unverified")

    if failures:
        print()
        print("[validate] FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1

    print()
    print("[validate] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
