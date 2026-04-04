# Parameter Golf — Competitive Strategy

> **27 days to deadline (April 30, 2026)**
> **Budget: $25 RunPod credits (~7 full runs) + pending grant application**

## CRITICAL: The Leaderboard is STALE (as of April 3, 2026)

The merged SOTA on the README is 1.1147 BPB, but **12+ pending PRs beat it**. The real competitive frontier:

| Tier | BPB | PR | Method |
|------|-----|-----|--------|
| SLOT aggressive (invalid) | 0.6951 | #1319 | SLOT-64 (EXCEEDS eval time limit) |
| **SLOT-32 (REAL TARGET)** | **0.7736** | **#1278** | **SLOT-32 + depth recurrence w/ iter_embed, 3-seed** |
| SLOT-24 | 0.8637 | #1313 | SLOT-24 AdamW, 3-seed validated |
| TTT + L-BFGS SLOT | 1.0096 | #1318 | L-BFGS logit-space SLOT, 3-seed |
| Causal SLOT + Pre-quant TTT | 1.0846 | #1306 | -0.051 nats vs merged SOTA |
| **Sliding only (no SLOT/TTT)** | **1.0897** | **#1296** | **SP4096 + Depth Recurrence + MuonEq-R** |
| N-gram matching | 1.1143 | #1309 | Exact n-gram on SOTA base |
| Merged SOTA | 1.1147 | #1019 | AR Self-Gen GPTQ + XSA-all |

**SLOT scaling law:** SLOT-8→~1.10, SLOT-16→~0.93, SLOT-24→~0.86, SLOT-32→~0.77

**CAUTION:** PRs #1272 and #1240 argue standard SLOT violates causal dependence — the delta is optimized on future tokens then applied to all positions. If SLOT is ruled illegal, scores revert to ~1.08-1.10.

**Our target:**
- To get on the leaderboard with sliding-only: need ~1.08 BPB
- To get on with SLOT: need ~0.85 BPB (easier with SLOT but more controversial)
- Safest strategy: strong model (~1.09) + SLOT (~0.85–0.95)

## Merged SOTA Stack (1.1147 BPB)

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

### Phase 4: Advanced Stack (~1.12→1.09 BPB) — Days 15–20
- [ ] GPTQ-lite (per-row clip search)
- [ ] Full Hessian GPTQ with self-gen calibration (damp=0.005)
- [ ] Parallel Muon + parameter banking (throughput)
- [ ] VE128 on deep layers
- [ ] Layerwise LN scale
- [ ] LZMA compression (vs zstd-22)
- [ ] SP4096 tokenizer (larger vocab — PR #1296 gets 1.0897 with this)
- [ ] Depth recurrence (shared layers, 2 passes — PR #1289, #1296)
- [ ] MuonEq-R (row-normalized Muon variant)
- [ ] QK-Gain 4.0–5.0
- [ ] Focal loss (gamma=1.0)
- **Compute: ~$30–50** (mixed 1×/8× runs)

### Phase 5: Eval-Time Optimization (~1.09→0.85–1.0 BPB) — Days 21–25
This is where the real competitive edge is now:
- [ ] Pre-quant TTT (fine-tune on val BEFORE GPTQ, -0.022 BPB)
- [ ] SLOT implementation (L-BFGS logit-space variant — PR #1318 approach)
- [ ] Causal SLOT variant (backward-looking only — PR #1306)
- [ ] Warmstart alpha tuning (0.85 default, inherit prev window delta)
- [ ] SLOT step/LR sweep within 600s eval budget
- [ ] Must fit in 600s eval time — budget carefully
- **Compute: ~$20–40** (8×H100 runs for eval timing)

### Phase 6: Final Submission — Days 26–27
- [ ] 3+ seed runs for statistical significance (p < 0.01)
- [ ] Prepare PR: README, submission.json, train logs, train_gpt.py
- [ ] Verify artifact < 16,000,000 bytes
- [ ] Verify eval time < 600s
- **Compute: ~$15–25** (3–7 final 8×H100 runs)

## Key PRs to Study (Updated April 3, 2026)

### Merged Records
| PR | Technique | Author | BPB |
|----|-----------|--------|-----|
| #1019 | Merged SOTA: Self-Gen GPTQ + XSA | abaybektursun | 1.1147 |
| #549 | LeakyReLU² + TTT + Parallel Muon | abaybektursun | 1.1194 |
| #374 | GPTQ-lite + EMA + warmdown | signalrush | 1.1228 |
| #287 | Partial RoPE + LN Scale + EMA + XSA | jfprincz | 1.1248 |
| #198 | XSA + EMA + Int6 MLP3x | jfprincz | 1.1271 |

### Pending PRs (The Real Frontier)
| PR | Technique | Author | BPB | Eval Method |
|----|-----------|--------|-----|-------------|
| #1319 | SLOT-64 warmstart | canivel | 0.6951 | SLOT (over time limit!) |
| #1313 | SLOT-24 aggressive | anthony-maio | 0.8637 | SLOT |
| #1318 | L-BFGS SLOT + TTT | renqianluo | 1.0096 | TTT + SLOT |
| #1306 | Causal SLOT + Pre-quant TTT | resouer | 1.0846 | SLOT + TTT |
| #1296 | SP4096 + Depth Recurrence | aryanbhosale | 1.0897 | Sliding only |
| #1287 | Vocab4096 + MLP4.0x | dentity007 | 1.1048 | Sliding only |
| #1309 | N-gram matching on SOTA | cadenmcmann | 1.1143 | N-gram |
| #1289 | PROTEUS v1.6 + Scylla | MatoTeziTanka | 1.0819 | TTT |

### Foundation Technique PRs
| PR | Technique |
|----|-----------|
| #65 | SmearGate |
| #162 | BigramHash |
| #286 | Late QAT with STE |
| #478 | XSA (cross-sequence attention) |
| #493 | LeakyReLU(0.5)² |
| #1176/#1229 | SLOT concept (arXiv:2505.12392v2) |
| #1006 | Pre-quant TTT concept |

## Decision Principles

1. **Stack orthogonal improvements** — each technique should help independently
2. **Measure everything** — no change without A/B comparison (3 seeds minimum)
3. **Artifact size is the binding constraint** — every byte matters
4. **SLOT is the new meta** — eval-time optimization is where the biggest gains are now
5. **1ms/step ≈ 0.006 BPB** — throughput is a first-class metric
6. **Study winning PRs** — don't reinvent what's proven
7. **Budget compute wisely** — develop on 1×H100, validate on 8×H100
8. **Respect the 600s eval limit** — SLOT must fit within budget
9. **Ship early, iterate** — a leaderboard entry that isn't SOTA still counts
