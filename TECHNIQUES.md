# Parameter Golf — Technique Catalog

Ranked by **BPB improvement per implementation effort**. Every serious submission should implement Rank 1–7 at minimum.

## Tier 1: Exceptional ROI

### 1. Sliding Window Evaluation
- **BPB gain:** -0.032 (largest single technique)
- **Effort:** Very Low
- **What:** Overlapping eval windows at stride=64 on 1024-token context. Every token scored with near-maximum context (~960+ tokens) instead of ~512 average.
- **Cost:** Zero training overhead. Eval takes longer but well within 10-min eval budget.
- **Orthogonal:** Yes — universal foundation, stacks with everything.

### 2. LeakyReLU(0.5)-squared Activation
- **BPB gain:** -0.003
- **Effort:** Very Low (1 line)
- **What:** Replace `relu(x)**2` with `leaky_relu(x, 0.5)**2`. Preserves negative gradient flow, eliminates dead neurons.
- **Orthogonal:** Fully orthogonal.
- **Ref:** PR #493

### 3. XSA — Exclusive Self-Attention
- **BPB gain:** -0.005+ (on all layers)
- **Effort:** Low (~2ms overhead)
- **What:** Subtract each token's own value vector from attention output. Forces tokens to attend to contextually novel information. Zero parameters. Apply to ALL layers (not just last 4).
- **Orthogonal:** Yes. Essential partner for EMA.
- **Ref:** PR #478

## Tier 2: Very High ROI

### 4. Int6 QAT + zstd-22 Compression
- **BPB gain:** -0.03 (indirect — frees space for more params)
- **Effort:** Medium
- **What:** STE fake-quantize 2D weights to 6-bit during training. Compress with zstandard-22 instead of zlib-9 (saves ~580KB). The freed space funds extra layers and wider MLP.
- **Late QAT:** Activate STE when LR scale drops below 0.15.
- **Orthogonal:** Foundation technique — everything builds on the space it frees.
- **Ref:** PR #286

### 5. EMA Weight Averaging (decay=0.997)
- **BPB gain:** -0.003 vs SWA
- **Effort:** Low
- **What:** Shadow parameter buffer updated every step. Replaces periodic SWA checkpoint collection.
- **Note:** Requires XSA to be effective — fails independently.
- **Ref:** PR #401

### 6. 11 Layers + MLP 3× Width
- **BPB gain:** -0.003 to -0.005
- **Effort:** Very Low (config change)
- **What:** Go from 9→11 layers and 2×→3× MLP (1536 hidden). Funded by int6 QAT space savings.
- **Orthogonal:** Yes.

### 7. Orthogonal Initialization
- **BPB gain:** Prerequisite enabler (modest standalone)
- **Effort:** Very Low (one line per layer)
- **What:** `nn.init.orthogonal_()` on all weight matrices. Required for SmearGate and BigramHash to work.
- **Orthogonal:** Yes.

## Tier 3: High ROI

### 8. FP16 Tied Embeddings
- **BPB gain:** -0.007 (reduces quant degradation from 0.007 to 0.0005)
- **Effort:** Very Low (one line in quantization function)
- **What:** Keep tied embedding/LM-head weight in FP16 instead of int8. Costs ~500KB — shrink MLP from 1024→992 to compensate if needed at int8 (or free under int6).
- **Ref:** 2026-03-18_FP16Embed_WD3600

### 9. U-Net Skip Connections
- **BPB gain:** -0.001 to -0.002
- **Effort:** Low
- **What:** Split layers into encoder (first N/2) and decoder (rest). Learned scalar skip weights connect corresponding layers.
- **Already in baseline.** Just keep it.

### 10. Warmdown Tuning
- **BPB gain:** Free convergence improvement
- **Effort:** Very Low (config)
- **What:** Match warmdown length to actual step count. Baseline uses 1200; winners use 3500–4000. Also tune `MATRIX_LR` (0.04→0.06).
- **Ref:** PR #364

## Tier 4: Moderate ROI

### 11. BigramHash Embeddings (3072×112)
- **BPB gain:** -0.001 to -0.002
- **Effort:** Medium
- **What:** Hash-based bigram embedding table (3072 buckets, 112 dims, projected to model dim). Increases effective vocabulary without proportional parameter cost.
- **Requires:** Orthogonal init.
- **Ref:** PR #162

### 12. SmearGate
- **BPB gain:** -0.001
- **Effort:** Medium
- **What:** Learned gate blending current and previous token embeddings for position-aware mixing.
- **Requires:** Orthogonal init.
- **Ref:** PR #65

### 13. GPTQ Post-Training Quantization
- **BPB gain:** -0.0006 (lite) to -0.002+ (full Hessian)
- **Effort:** Low–Medium
- **What:** Per-row clip percentile search (lite) or full Hessian with Cholesky error compensation (full). Applied after training.
- **Full Hessian** requires calibration data → use AR self-generation.
- **Ref:** PR #535, #609

### 14. AR Self-Generated GPTQ Calibration
- **BPB gain:** Part of -0.005 SOTA improvement
- **Effort:** Medium
- **What:** Model generates 64 sequences × 2048 tokens at temp=0.8 for GPTQ calibration. Avoids accessing train/val data during quantization.
- **Ref:** PR #1019

### 15. Partial RoPE (16/64 dims)
- **BPB gain:** ~-0.001
- **Effort:** Low
- **What:** Apply rotary embeddings to only 16 of 64 head dimensions.
- **Ref:** PR #315

### 16. Parallel Muon + Parameter Banking
- **BPB gain:** Throughput only (more steps in 10 min)
- **Effort:** Medium–High
- **What:** Batch 66 linear weights into 4 contiguous banks for batched Newton-Schulz. Reduces step time to ~83ms.
- **Ref:** PR #399

## Tier 5: Low ROI / Risky

### 17. Test-Time Training (TTT)
- **BPB gain:** -0.002 (unreliable — neutral/negative on latest stack)
- **Effort:** High
- **What:** LoRA-based SGD adaptation on already-scored val tokens. Score-first protocol.
- **Warning:** 25+ failed attempts on the SOTA stack. Only works on earlier architectures.
- **Ref:** PR #549

### 18. Ternary/1-bit Quantization
- **BPB gain:** Underperforms int6 stack significantly
- **Effort:** Very High
- **What:** 73M–106M params quantized to {-1, 0, 1}.
- **Better suited for unlimited compute track.**

## What Definitely Does NOT Work

| Technique | Result | Why |
|-----------|--------|-----|
| MoE (small scale) | -0.06 to -0.08 worse | Routing overhead, not enough params |
| Int4 quantization | Catastrophic | 10× worse than int5 |
| Knowledge distillation | +0.09 to +0.41 worse | I/O overhead kills training steps |
| 12L at seq2048 | Too slow | ~5,590 steps vs ~9,000 for 11L |
| N-gram eval caches | Invalidated | Probability normalization bugs |
| CDQuant, Qronos | Fail | Advanced quant beyond GPTQ doesn't help |

## Tier 6: Eval-Time Optimization (The New Meta — April 2026)

These techniques have **redefined the competition**. The pending PRs show SLOT-based entries pushing below 1.0 BPB.

### 19. SLOT — Score-First Learned Optimization at Test-time
- **BPB gain:** -0.10 to -0.44 (depending on steps/config)
- **Effort:** Medium–High
- **What:** Per sliding window, optimize throwaway parameters (delta + logit_bias) against validation loss using frozen model weights. Only tokens already scored contribute to the optimization (causal/score-first protocol).
- **Variants:**
  - **AdamW SLOT:** 16–64 steps per window, delta in hidden space (512-dim) + logit_bias (1024-dim). PR #1313 achieves 0.8637 BPB.
  - **L-BFGS SLOT:** Logit-space delta (1024-dim), L-BFGS with warm-start. PR #1318 achieves 1.0096 BPB. More stable, fewer hyperparams.
  - **Causal SLOT:** Only backward-looking positions used for optimization. PR #1306 achieves 1.0846 BPB.
- **Warmstart:** Inherit 85% of previous window's delta — exploits sequential locality.
- **Eval time budget:** 230–825s depending on config. Must fit in 600s eval limit.
- **Key insight:** 1,536 free parameters per window against ~96 scored tokens = 16 params per token.
- **Ref:** arXiv:2505.12392v2, PRs #1176, #1229, #1306, #1313, #1318, #1319

### 20. Pre-Quant TTT (Test-Time Training)
- **BPB gain:** -0.022
- **Effort:** Medium
- **What:** Fine-tune EMA model on validation data BEFORE GPTQ quantization (not after — post-quant TTT fails on GPTQ stacks). AdamW lr=0.0005, cosine decay, 6 epochs, freeze first 2 blocks, grad clip 1.0. Takes ~111s.
- **Key insight:** Post-quant TTT was tried 25+ times and failed (PR #756). Pre-quant TTT works because adapted weights quantize better.
- **Ref:** PR #1006, #1306

### 21. Additional New Techniques (April 2026 PRs)
- **MTP (Multi-Token Prediction):** 2 auxiliary heads, loss weight 0.1 — used in PR #1318
- **Focal loss (gamma=1.0):** Downweight easy tokens — used in PR #1319
- **Sqrt warmdown:** `1 - sqrt(1 - frac)` curve, more time at moderate LR — PR #1319
- **MuonEq-R:** Row-normalized Muon variant — PRs #1289, #1296
- **Depth Recurrence:** Shared layer weights, 2 forward passes — PRs #1289, #1296
- **Scylla tokenizer:** TokenMonster-derived 998-token vocab — PR #1289
- **SP4096:** Larger SentencePiece vocab — PRs #1287, #1296 (1.0897 without SLOT!)
- **N-gram sequence matching:** Exact n-gram matching as eval-time boost — PR #1309 (1.1143)
- **QK-Gain 4.0–5.0:** Higher initial query scaling — PRs #1296, #1303, #1313

## Competitive Landscape (April 3, 2026)

The leaderboard README is **stale**. Real competitive state from open PRs:

| Tier | Best BPB | Method | Key Technique |
|------|----------|--------|---------------|
| SLOT aggressive | 0.6951 | PR #1319 | SLOT-64 warmstart (OVER eval time limit) |
| SLOT moderate | 0.8637 | PR #1313 | SLOT-24 AdamW |
| TTT + SLOT | 1.0096 | PR #1318 | L-BFGS SLOT + TTT |
| Causal SLOT + TTT | 1.0846 | PR #1306 | Causal SLOT + Pre-quant TTT |
| Sliding only (no SLOT) | 1.0897 | PR #1296 | SP4096 + Depth Recurrence + MuonEq-R |
| Merged SOTA | 1.1147 | PR #1019 | AR Self-Gen GPTQ + XSA-all |

## Implementation Priority (Recommended Order)

**Phase A — Model Quality (sliding window eval)**
1. Reproduce baseline (1.2244)
2. Sliding window eval (→ ~1.19)
3. LeakyReLU² + warmdown tuning (→ ~1.18)
4. Int6 QAT + zstd-22 + 11L + MLP 3× (→ ~1.15)
5. XSA all layers + EMA (→ ~1.13)
6. BigramHash + SmearGate + OrthoInit (→ ~1.12)
7. GPTQ + self-gen calibration (→ ~1.11)
8. SP4096 + Depth Recurrence + MuonEq-R (→ ~1.09)

**Phase B — Eval-Time Optimization (SLOT/TTT on top)**
9. Pre-quant TTT (→ ~1.07)
10. SLOT (L-BFGS or AdamW variant) (→ ~0.9–1.0)
11. Tune SLOT hyperparams within 600s eval budget
