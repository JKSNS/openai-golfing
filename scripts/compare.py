#!/usr/bin/env python3
"""Compare quantized_ttt val_bpb across two or more logs with proper stats.

Welch's t-test, Cohen's d, 95% CI on the delta. Supports glob patterns.

Usage:
  python3 compare.py
  python3 compare.py logs/repro_seed42.log logs/cwttt_seed42_floor0.05.log
  python3 compare.py --baseline 'logs/repro_seed*.log' --treatment 'logs/cwttt_*.log'
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import sys
from pathlib import Path


VAL_BPB = re.compile(r"quantized_ttt[^\n]*val_bpb[\s:=]+(\d+\.\d+)")
SOTA = 1.0810


def extract(path: Path) -> float | None:
    text = path.read_text(errors="replace")
    m = VAL_BPB.findall(text)
    return float(m[-1]) if m else None


def expand(patterns: list[str]) -> list[Path]:
    out = []
    for p in patterns:
        hits = sorted(Path(x) for x in glob.glob(p))
        out.extend(hits if hits else [Path(p)])
    return out


def stats(vals: list[float]) -> tuple[float, float, float, int]:
    n = len(vals)
    if n == 0:
        return float("nan"), 0.0, 0.0, 0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
    sd = math.sqrt(var)
    sem = sd / math.sqrt(n) if n > 0 else 0.0
    return mean, sd, sem, n


def welch(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Returns (delta, t, df). delta = mean(a) - mean(b). df via Welch-Satterthwaite."""
    ma, sa, _, na = stats(a)
    mb, sb, _, nb = stats(b)
    if na < 2 or nb < 2:
        return ma - mb, float("nan"), float("nan")
    va, vb = sa * sa, sb * sb
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return ma - mb, float("inf"), float("nan")
    t = (ma - mb) / se
    num = (va / na + vb / nb) ** 2
    den = (va * va) / (na * na * (na - 1)) + (vb * vb) / (nb * nb * (nb - 1))
    df = num / den if den > 0 else float("nan")
    return ma - mb, t, df


def cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, sa, _, na = stats(a)
    mb, sb, _, nb = stats(b)
    pooled = math.sqrt(((na - 1) * sa * sa + (nb - 1) * sb * sb) / (na + nb - 2))
    return (ma - mb) / pooled if pooled > 0 else float("nan")


def t_critical(df: float, alpha: float = 0.01) -> float:
    """Two-sided critical t for p<alpha. Lookup with 3.25 fallback for df>=10."""
    table = {1: 63.66, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032, 6: 3.707,
             7: 3.499, 8: 3.355, 9: 3.250, 10: 3.169, 15: 2.947, 20: 2.845}
    if math.isnan(df):
        return float("nan")
    df_int = int(df)
    if df_int <= 1: return table[1]
    if df_int >= 20: return 2.845
    return table.get(df_int, 3.25)


def render(rows: list[tuple[Path, float | None]], label: str) -> list[float]:
    print(f"\n[{label}]")
    print(f"{'log':<60s} {'val_bpb':>9s}")
    print("-" * 71)
    vals = []
    for p, v in rows:
        if v is None:
            print(f"{p.name:<60s} {'(no val_bpb)':>9s}")
        else:
            print(f"{p.name:<60s} {v:>9.5f}")
            vals.append(v)
    if vals:
        m, sd, sem, n = stats(vals)
        print(f"  n={n}  mean={m:.5f}  sd={sd:.5f}  sem={sem:.5f}")
    return vals


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="*")
    p.add_argument("--baseline", action="append", default=[])
    p.add_argument("--treatment", action="append", default=[])
    args = p.parse_args()

    if not args.logs and not args.baseline:
        args.logs = ["/workspace/logs/*.log"]

    if args.baseline or args.treatment:
        base = [(p, extract(p)) for p in expand(args.baseline)]
        treat = [(p, extract(p)) for p in expand(args.treatment)]
        a = render(base, "baseline")
        b = render(treat, "treatment")
        if not a or not b:
            return 1
        delta, t, df = welch(b, a)   # treatment - baseline
        d = cohens_d(b, a)
        tc = t_critical(df, 0.01)
        margin = abs(t * 0) + (abs(delta - (delta if math.isnan(t) else (t - tc) * (delta / t))) if t else 0)
        # simpler CI: delta +/- t_crit * SE
        ma, sa, _, na = stats(b); mb, sb, _, nb = stats(a)
        if na >= 2 and nb >= 2:
            se = math.sqrt(sa * sa / na + sb * sb / nb)
            ci = tc * se
        else:
            ci = float("nan")
        print()
        print("[Welch contrast: treatment - baseline]")
        print(f"  delta:       {delta:+.5f}")
        print(f"  Cohen's d:   {d:+.3f}")
        print(f"  Welch t:     {t:+.3f}  (df={df:.1f})")
        print(f"  99% CI:      [{delta - ci:+.5f}, {delta + ci:+.5f}]")
        record_target = SOTA - 0.005
        if not math.isnan(ci) and (ma + ci) < record_target:
            print(f"  RECORD: upper-CI of treatment ({ma+ci:.5f}) beats target ({record_target}) at p<0.01")
        elif ma < record_target:
            print(f"  CLOSE: treatment mean ({ma:.5f}) beats target but CI does not.")
        else:
            print(f"  NOT RECORD: treatment mean ({ma:.5f}) does not beat target ({record_target}).")
        return 0

    paths = expand(args.logs)
    rows = [(p, extract(p)) for p in paths]
    print(f"{'log':<60s} {'val_bpb':>9s} {'vs SOTA':>9s}")
    print("-" * 81)
    for p, v in rows:
        if v is None:
            print(f"{p.name:<60s} {'(no val_bpb)':>9s}")
        else:
            print(f"{p.name:<60s} {v:>9.5f} {v - SOTA:+9.5f}")

    valid = [(p, v) for p, v in rows if v is not None]
    if len(valid) >= 2:
        repro = next((v for p, v in valid if "repro" in p.name.lower()), None)
        cwttt = next((v for p, v in valid if "cwttt" in p.name.lower()), None)
        if repro is not None and cwttt is not None:
            d = cwttt - repro
            print(f"\n  CW-TTT delta vs repro: {d:+.5f}")
            if d <= -0.001:
                print("  KEEP GOING: CW-TTT improves by >=0.001. Run scripts/run_phase3a.sh for floor sweep.")
            elif d >= 0.001:
                print("  DROP: CW-TTT regresses by >=0.001. Fall back to clean reproduction submission.")
            else:
                print("  NOISE: delta within +/- 0.001. Run a second seed before deciding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
