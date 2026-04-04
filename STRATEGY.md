# Parameter Golf — Competitive Strategy

> **27 days to deadline (April 30, 2026)**
> **Budget: $25 RunPod credits (~7 full runs) + pending grant application**

## Current SOTA Stack (1.1147 BPB)

The #1 entry (abaybektursun, PR #1019) stacks ~20 orthogonal techniques built across dozens of community PRs:

```
Architecture:   11L, 512dim, 8Q/4KV heads, MLP 3× (1536), LeakyReLU(0.5)²
Embeddings:     Tied FP16, BigramHash 3072×112, SmearGate
Attention:      XSA all 11 layers, Partial RoPE 16/64, LN scale 1/sqrt(L+1)
Structure:      U-Net skips, VE128 on layers 9-10
Optimizer:      Parallel Muon + Parameter Banking, EMA(0.997) + Tight SWA(50)
Training:       Late QAT (STE at LR<0.15), warmdown=4000, ~6920 steps @ 86.7ms
Quantization:   Full Hessian GPTQ int6, AR self-gen calibration (64×2048 @ temp=0.8)
Compression:    LZMA preset=9, artifact ~15.9MB
Eval:           Sliding window (stride=64)
```

## Our Phased Plan

### Phase 1: Reproduce Baseline (~1.22 BPB) — Days 1–2
- [ ] Clone upstream `train_gpt.py` into `src/`
- [ ] Download data (`--variant sp1024 --train-shards 10`)
- [ ] Run baseline on available hardware, confirm ~1.22 BPB
- [ ] Understand every component of the 1126-line script
- **Compute: $0** (local or existing GPU)

### Phase 2: Quick Wins (~1.19→1.15 BPB) — Days 3–6
Highest ROI techniques, minimal implementation:
- [ ] Sliding window eval (stride=64) → -0.032 BPB
- [ ] LeakyReLU(0.5)² activation → -0.003 BPB
- [ ] Warmdown 1200→3600, matrix_lr 0.04→0.06
- [ ] FP16 tied embeddings (skip int8 on embedding tensor)
- [ ] Int6 QAT + zstd-22 compression
- [ ] 11 layers + MLP 3× (1536 hidden)
- [ ] Orthogonal initialization
- **Compute: ~$10** (1×H100, ~4 hrs)

### Phase 3: Core Stack (~1.15→1.12 BPB) — Days 7–14
Medium-effort techniques with proven impact:
- [ ] XSA (exclusive self-attention) on all layers
- [ ] EMA(0.997) replacing SWA
- [ ] BigramHash embeddings (start at 2048, tune size)
- [ ] SmearGate position-mixing
- [ ] Partial RoPE (16/64 dims)
- [ ] Muon weight decay tuning
- [ ] Late QAT timing optimization
- **Compute: ~$30–50** (1×H100 iteration + 2–3 8×H100 validation runs)

### Phase 4: Advanced Stack (~1.12→1.11 BPB) — Days 15–22
- [ ] GPTQ-lite (per-row clip search)
- [ ] Full Hessian GPTQ with self-gen calibration
- [ ] Parallel Muon + parameter banking (throughput)
- [ ] VE128 on deep layers
- [ ] Layerwise LN scale
- [ ] LZMA compression (vs zstd-22)
- **Compute: ~$30–50** (mixed 1×/8× runs)

### Phase 5: Novel Ideas & Submission — Days 23–27
- [ ] Explore one novel direction (see below)
- [ ] Final 5–10 runs for statistical significance
- [ ] Prepare PR: README, submission.json, train logs, train_gpt.py
- **Compute: ~$20–40** (8×H100 final runs)

## Novel Directions Worth Exploring

These could differentiate us from the pack:

| Idea | Potential | Risk | Notes |
|------|-----------|------|-------|
| Depth recurrence | High | Medium | Reuse layer weights → more "virtual" depth in 16MB |
| State-space hybrid | High | High | Mamba-style layers for some positions — on OpenAI wishlist |
| Progressive training | Medium | Medium | Start small, grow model during training |
| Learned compression | Medium | Low | Optimize model weights for compressibility |
| Custom CUDA kernels | Medium | High | Fused attention+MLP could save ~2ms/step |
| Better tokenizer | Low | Very High | Score verification is extremely strict |

## Key PRs to Study

| PR | Technique | Author | BPB |
|----|-----------|--------|-----|
| #1019 | SOTA: Self-Gen GPTQ + XSA | abaybektursun | 1.1147 |
| #549 | LeakyReLU² + TTT + Parallel Muon | abaybektursun | 1.1194 |
| #414 | Base stack for #549 | abaybektursun | — |
| #374 | GPTQ-lite + EMA + warmdown | signalrush | 1.1228 |
| #287 | Partial RoPE + LN Scale + EMA + XSA | jfprincz | 1.1248 |
| #198 | XSA + EMA + Int6 MLP3x | jfprincz | 1.1271 |
| #65 | SmearGate | — | — |
| #162 | BigramHash | — | — |
| #286 | Late QAT | — | — |
| #478 | XSA concept | gowtham0992 | — |
| #493 | LeakyReLU² | — | — |

## Decision Principles

1. **Stack orthogonal improvements** — each technique should help independently
2. **Measure everything** — no change without A/B comparison (3 seeds minimum)
3. **Artifact size is the binding constraint** — every byte matters
4. **Eval tricks are free** — sliding window, XSA don't cost training time
5. **1ms/step ≈ 0.006 BPB** — throughput is a first-class metric
6. **Study winning PRs** — don't reinvent what's proven
7. **Budget compute wisely** — develop on 1×H100, validate on 8×H100
8. **Ship early, iterate** — a leaderboard entry that isn't SOTA still counts
