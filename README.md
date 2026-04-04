# OpenAI Parameter Golf — Our Entry

> ## DEADLINE: April 30, 2026 ⏰
> 27 days remaining as of April 3, 2026

Our workspace for the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) challenge.

**Goal:** Train the best language model that fits in a **16MB artifact** and trains in **under 10 minutes on 8×H100 SXM GPUs**, scored by **bits-per-byte (BPB)** on the FineWeb validation set.

## Scoreboard Context

| Entry | BPB | Notes |
|-------|-----|-------|
| Naive Baseline | 1.2244 | 9L 512dim 1024vocab |
| Merged SOTA | 1.1147 | abaybektursun, Mar 25 — PR #1019 |
| Best pending (sliding only) | 1.0897 | PR #1296 — SP4096 + Depth Recurrence |
| **Best pending (SLOT, 3-seed)** | **0.7736** | **PR #1278 — SLOT-32 + depth recurrence** |
| Best pending (SLOT) | 0.8637 | PR #1313 — SLOT-24 eval-time optimization |
| Best pending (TTT+SLOT) | 1.0096 | PR #1318 — L-BFGS SLOT + TTT |
| **Our target (sliding)** | **≤1.08** | **Competitive model quality** |
| **Our target (L-BFGS SLOT)** | **≤0.73** | **World record — must beat PR #1278 (0.7736)** |

## Quick Start

```bash
# 1. Clone upstream reference
git clone https://github.com/openai/parameter-golf.git upstream

# 2. Download data (full val + 10 training shards for iteration)
python3 upstream/data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# 3. Quick smoke test (1 GPU)
RUN_ID=smoke \
DATA_PATH=./upstream/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./upstream/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 src/train_gpt.py

# 4. Full submission run (8×H100)
RUN_ID=submission \
DATA_PATH=./upstream/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./upstream/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 src/train_gpt.py
```

## Repo Structure

```
├── CLAUDE.md          # Agent context — rules, architecture, commands
├── STRATEGY.md        # Phased experiment plan ranked by ROI
├── TECHNIQUES.md      # Detailed reference on every proven technique
├── FUNDING.md         # Compute grant application materials
├── README.md          # This file
├── upstream/          # Clone of openai/parameter-golf (reference only)
├── src/               # Our training script(s)
│   └── train_gpt.py   # Modified training script
├── experiments/       # Experiment logs and results
├── records/           # Submission-ready artifacts
│   └── our_submission/
└── scripts/           # Helper scripts
```

## Key Rules

- **16,000,000 bytes** max artifact (code + compressed model) — decimal MB, not MiB
- **10 minutes** training on 8×H100 SXM
- **10 minutes** evaluation on 8×H100 SXM
- **No network** during evaluation — fully self-contained
- **No validation data** during training
- **New SOTA must beat existing by ≥0.005 nats** at p < 0.01
- Test-time training allowed only on already-evaluated tokens

## Compute Budget

We have **$25 in RunPod credits** (~7 full 8×H100 runs). See [FUNDING.md](FUNDING.md) for grant application materials to request more.

## Strategy

See [STRATEGY.md](STRATEGY.md) for our phased approach and [TECHNIQUES.md](TECHNIQUES.md) for the full technique catalog.

## Key Links

- [Challenge Repo](https://github.com/openai/parameter-golf)
- [Compute Grant Form](https://openai.com/index/parameter-golf/#credit-form)
- [RunPod Template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th)
- [Discord](https://discord.gg/openai) — #parameter-golf-discussions
- [Unofficial Leaderboard](https://parameter-golf.github.io/)
