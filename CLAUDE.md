# Parameter Golf — Agent Context

## What This Is

This repo is our workspace for the **OpenAI Parameter Golf** challenge.
Train the best language model fitting in **16MB**, trainable in **≤10 min on 8×H100 SXM**, scored by **bits per byte (BPB)** on FineWeb validation.

- Upstream repo: https://github.com/openai/parameter-golf
- **Deadline: April 30, 2026**
- Merged SOTA: **1.1147 BPB** (abaybektursun, PR #1019, Mar 25)
- **Real frontier (pending PRs, Apr 3):** ~0.86–1.09 BPB (SLOT-based eval optimization)
- Best pending sliding-only: **1.0897 BPB** (PR #1296, SP4096 + depth recurrence)
- Naive baseline: **1.2244 BPB**

## Hard Constraints

| Constraint | Limit |
|---|---|
| Artifact size | ≤ 16,000,000 bytes (code + compressed model, decimal MB) |
| Training time | ≤ 600 seconds on 8×H100 SXM |
| Evaluation time | ≤ 600 seconds on 8×H100 SXM |
| Network during eval | **None** — fully self-contained |
| Validation data during training | **Forbidden** |
| Test-time training | Only on tokens already evaluated |
| New SOTA threshold | Must beat existing by ≥ 0.005 nats at p < 0.01 |

## Scoring

- `val_bpb` = bits per byte on FineWeb validation set (tokenizer-agnostic)
- Computed as: `(cross_entropy_loss / ln(2)) * (tokens / bytes)`
- Lower is better
- Byte counting uses SentencePiece lookup tables with leading-space handling

## Baseline Architecture (train_gpt.py, 1126 lines)

```
Layers:         9 (4 encoder + 5 decoder, U-Net skips)
Model dim:      512
Heads:          8 query, 4 KV (GQA)
MLP:            2× width (1024), ReluSquared activation
Vocab:          1024 (SentencePiece BPE)
Embeddings:     Tied (input = output projection)
Normalization:  RMSNorm (no learned params)
Position:       RoPE (base=10000)
Logit cap:      softcap=30.0
Optimizer:      Muon (matrix params) + Adam (scalars, embeddings)
LR:             matrix=0.04, embed=0.05, scalar=0.04
Warmdown:       1200 iters (wallclock-adaptive)
Batch:          524,288 tokens/step
Seq length:     1024
Quantization:   int8 per-row + zlib-9
~Steps in 10m:  ~13,700
```

## The "Standard Stack" (what every competitive entry uses)

These techniques collectively move from 1.2244 → ~1.12 BPB:

1. **Sliding Window Eval** — stride=64 on 1024 context, -0.032 BPB, zero training cost
2. **Int6 QAT** — STE fake-quantize during training, frees ~25% artifact space
3. **zstd-22 compression** — replaces zlib-9, saves ~580KB
4. **11 layers** (from 9) — funded by quantization savings
5. **MLP 3×** (1536 hidden) — wider FFN
6. **LeakyReLU(0.5)²** — one-line activation swap, -0.003 BPB
7. **XSA all layers** — exclusive self-attention, zero params, -0.005+ BPB
8. **EMA(0.997)** — replaces SWA, -0.003 BPB vs SWA
9. **BigramHash(3072×112)** — bigram hash embeddings
10. **GPTQ post-quant** — per-row clip search or full Hessian
11. **AR self-gen calibration** — model generates own GPTQ calibration data
12. **Muon optimizer** with weight decay + momentum warmup
13. **Orthogonal init** — prerequisite for SmearGate/BigramHash
14. **SmearGate** — position-mixing gate
15. **U-Net skip connections** — encoder-decoder with learned skip weights
16. **Partial RoPE** — 16/64 dims only
17. **FP16 tied embeddings** — skip int8 quant on embedding tensor

## Critical Performance Rule

**Every 1ms of per-step overhead costs ~0.006 BPB** in lost training tokens. Techniques must deliver gains exceeding their throughput cost.

## What Does NOT Work

- MoE at small scale (-0.06 to -0.08 BPB worse)
- Int4 quantization (catastrophic degradation)
- Knowledge distillation (+0.09 to +0.41 worse, I/O overhead kills steps)
- 12 layers at seq2048 (too slow, ~5,590 steps vs ~9,000)
- N-gram eval caches (probability normalization issues)
- TTT on well-regularized models (neutral/negative on latest stack)

## Repo Structure

```
├── CLAUDE.md          # This file
├── STRATEGY.md        # Phased plan
├── TECHNIQUES.md      # Full technique catalog
├── FUNDING.md         # Grant application materials
├── README.md          # Project overview
├── upstream/          # openai/parameter-golf clone (reference only)
├── src/               # Our training script(s)
│   └── train_gpt.py
├── experiments/       # Experiment logs
├── records/           # Submission artifacts
└── scripts/           # Helpers
```

## Commands

```bash
# Download data
python3 upstream/data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Train 1 GPU (dev)
RUN_ID=test DATA_PATH=./upstream/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./upstream/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 torchrun --standalone --nproc_per_node=1 src/train_gpt.py

# Train 8 GPU (submission)
RUN_ID=full DATA_PATH=./upstream/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./upstream/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 torchrun --standalone --nproc_per_node=8 src/train_gpt.py

# Check artifact size
python3 -c "import os; print(f'{os.path.getsize(\"final_model.int8.ptz\") / 1e6:.2f} MB')"
```

## Agent Guidelines

- **Read upstream code first** — understand what baseline does before changing anything
- **One technique at a time** — isolate impact of each change
- **Always check artifact size** — must stay under 16,000,000 bytes
- **Log everything** — every experiment needs config + result recorded
- **Prioritize by ROI** — see TECHNIQUES.md ranking
- **Profile throughput** — ms/step matters as much as architecture
- **Don't reinvent** — study winning PRs, adapt proven implementations
- **Statistical significance** — run 3+ seeds, report mean and std
