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

## Implementation Priority (Recommended Order)

1. Reproduce baseline (1.2244)
2. Sliding window eval (→ ~1.19)
3. LeakyReLU² + warmdown tuning (→ ~1.18)
4. Int6 QAT + zstd-22 + 11L + MLP 3× (→ ~1.15)
5. XSA all layers + EMA (→ ~1.13)
6. BigramHash + SmearGate + OrthoInit (→ ~1.12)
7. GPTQ + self-gen calibration (→ ~1.11)
8. Novel ideas (→ <1.11?)
