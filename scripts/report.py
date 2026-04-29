#!/usr/bin/env python3
"""Parse a run log into a structured report.

Captures everything a research-grade run needs: seed, env vars, GPU info,
step counts, train_time, eval_time, val_bpb at every checkpoint, score-first
compliance markers, artifact bytes, $ cost. Emits both JSON (for
machine-aggregation) and Markdown (for humans).

Usage:
  python3 scripts/report.py path/to/run.log
  python3 scripts/report.py path/to/run.log --out runs/cwttt_seed42/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


COST_PER_HOUR_USD = 25.0   # 8xH100 SXM RunPod approx
SOTA = 1.0810


PATTERNS = {
    "train_step": re.compile(r"^(\d+)/(\d+) train_loss:\s*([\d.]+) train_time:\s*([\d.]+)m tok/s:\s*(\d+)"),
    "val_step": re.compile(r"^(\d+)/(\d+) val_loss:\s*([\d.]+) val_bpb:\s*([\d.]+)"),
    "ttt_start": re.compile(r"ttt:start chunks=(\d+) ttt_lr=([\d.]+) ttt_epochs=(\d+)"),
    "eval_line": re.compile(r"^(\S+)\s+val_loss:([\d.]+) val_bpb:([\d.]+) eval_time:(\d+)ms"),
    "params": re.compile(r"model_params:(\d+)"),
    "peak_mem": re.compile(r"peak memory allocated:\s*(\d+)"),
    "training_done": re.compile(r"stopping_early:.*train_time:\s*(\d+)ms.*step:\s*(\d+)"),
    "bytes_total": re.compile(r"bytes_total:\s*(\d+)"),
}


def gpu_info() -> list[dict]:
    try:
        import torch
        return [
            {
                "idx": i,
                "name": torch.cuda.get_device_properties(i).name,
                "cc": f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}",
            }
            for i in range(torch.cuda.device_count())
        ]
    except Exception:
        return []


def git_sha() -> str | None:
    here = Path(__file__).resolve().parent.parent
    try:
        return subprocess.check_output(["git", "-C", str(here), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def grep_env_for_run(text: str) -> dict[str, str]:
    captured = {}
    for var in ("SEED", "QK_GAIN_INIT", "TTT_ENABLED", "TTT_LR", "TTT_EPOCHS",
                "TTT_CONFIDENCE_WEIGHTED", "TTT_WEIGHT_FLOOR", "VOCAB_SIZE",
                "DATA_DIR", "MAX_WALLCLOCK_SECONDS", "ITERATIONS"):
        v = os.environ.get(var)
        if v is not None:
            captured[var] = v
    return captured


def parse(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    out: dict = {
        "log_path": str(log_path),
        "log_bytes": len(text.encode()),
        "lines": text.count("\n"),
        "evals": [],
        "train_progress": [],
        "val_progress": [],
    }

    last_train = None
    for line in text.splitlines():
        m = PATTERNS["train_step"].match(line)
        if m:
            last_train = {
                "step": int(m.group(1)),
                "total": int(m.group(2)),
                "train_loss": float(m.group(3)),
                "train_time_min": float(m.group(4)),
                "tok_per_sec": int(m.group(5)),
            }
            out["train_progress"].append(last_train)
            continue
        m = PATTERNS["val_step"].match(line)
        if m:
            out["val_progress"].append({
                "step": int(m.group(1)),
                "val_loss": float(m.group(3)),
                "val_bpb": float(m.group(4)),
            })
            continue
        m = PATTERNS["eval_line"].match(line)
        if m:
            out["evals"].append({
                "label": m.group(1),
                "val_loss": float(m.group(2)),
                "val_bpb": float(m.group(3)),
                "eval_time_ms": int(m.group(4)),
            })
            continue
        for key in ("ttt_start", "params", "peak_mem", "training_done", "bytes_total"):
            m = PATTERNS[key].match(line) if key != "training_done" else PATTERNS[key].search(line)
            if m:
                out.setdefault(key, []).append(m.groups())

    out["last_train"] = last_train
    final = next((e for e in reversed(out["evals"]) if "ttt" in e["label"].lower()), None)
    out["final_eval"] = final
    if final:
        out["final_val_bpb"] = final["val_bpb"]
        out["delta_vs_sota"] = round(final["val_bpb"] - SOTA, 5)

    # cost = (train_seconds + total_eval_seconds) / 3600 * $25
    train_seconds = (last_train["train_time_min"] * 60) if last_train else 0
    eval_seconds = sum(e["eval_time_ms"] for e in out["evals"]) / 1000.0
    out["train_seconds"] = round(train_seconds, 1)
    out["eval_seconds"] = round(eval_seconds, 1)
    out["wall_seconds"] = round(train_seconds + eval_seconds, 1)
    out["cost_usd"] = round(out["wall_seconds"] / 3600 * COST_PER_HOUR_USD, 2)

    out["meta"] = {
        "host": socket.gethostname(),
        "git_sha": git_sha(),
        "gpu": gpu_info(),
        "captured_env": grep_env_for_run(text),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return out


def render_md(data: dict) -> str:
    f = data.get("final_eval") or {}
    delta = data.get("delta_vs_sota")
    delta_s = f"{delta:+.5f}" if delta is not None else "-"
    rows = [
        f"# Run report",
        "",
        f"- Log: `{data['log_path']}`",
        f"- Generated: {data['meta']['generated_at']}",
        f"- Git SHA: {data['meta']['git_sha']}",
        f"- Host: {data['meta']['host']}",
        f"- GPU: {', '.join(g['name'] for g in data['meta']['gpu']) or 'unknown'}",
        "",
        "## Score",
        "",
        f"| metric | value |",
        f"|---|---|",
        f"| final val_bpb (quantized_ttt) | {f.get('val_bpb','-')} |",
        f"| delta vs SOTA ({SOTA}) | {delta_s} |",
        f"| eval time (s) | {(f.get('eval_time_ms',0)/1000):.1f} |",
        "",
        "## Wallclock + cost",
        "",
        f"| metric | value |",
        f"|---|---|",
        f"| train seconds | {data['train_seconds']} |",
        f"| eval seconds | {data['eval_seconds']} |",
        f"| wall seconds | {data['wall_seconds']} |",
        f"| cost USD (@${COST_PER_HOUR_USD}/hr) | {data['cost_usd']} |",
        "",
        "## All evals",
        "",
        "| label | val_loss | val_bpb | eval_ms |",
        "|---|---|---|---|",
    ]
    for e in data.get("evals", []):
        rows.append(f"| {e['label']} | {e['val_loss']:.4f} | {e['val_bpb']:.5f} | {e['eval_time_ms']} |")
    rows += ["", "## Captured env", "", "```"]
    for k, v in data["meta"]["captured_env"].items():
        rows.append(f"{k}={v}")
    rows.append("```")
    if data.get("last_train"):
        lt = data["last_train"]
        rows += ["", "## Last train log line", "",
                 f"step {lt['step']}/{lt['total']}, loss {lt['train_loss']}, "
                 f"{lt['train_time_min']:.1f} min, {lt['tok_per_sec']} tok/s"]
    return "\n".join(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument("--out", type=Path, help="dir to write report.json + report.md")
    args = p.parse_args()

    if not args.log.exists():
        print(f"FAIL  log not found: {args.log}", file=sys.stderr)
        return 2

    data = parse(args.log)
    md = render_md(data)
    print(md)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(json.dumps(data, indent=2))
        (args.out / "report.md").write_text(md)
        print(f"\nwrote {args.out}/report.json and {args.out}/report.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
