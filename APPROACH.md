# Our Approach — Novel Submission Strategy

> **Target: Beat PR #1278 (0.7736 BPB) by a significant margin**

## Philosophy

Don't stack 30 techniques from other PRs. Instead:

1. **Fork the proven winner** (PR #1278, 1516 lines, 0.7736 BPB)
2. **Add ONE genuinely novel technique** that shifts the SLOT scaling curve
3. **Add 2-3 targeted improvements** (TTT, better optimizer params)
4. **Keep the code clean** (<2000 lines, readable, maintainable)

## Base: PR #1278 (proven 0.7736 BPB)

Stored at `src/train_gpt_pr1278_base.py`. Key properties:
- 11 physical layers → 13 virtual (depth recurrence on 4,5)
- iter_embed + iter_gate conditioning (novel in PR #1278)
- VRL (value residual learning)
- SLOT-32 AdamW with per-sample delta + logit bias
- GPTQ int6, LZMA compression
- 1516 clean lines

## Novel Technique: SLOT-Aware Meta-Training (FOMAML)

**The key insight nobody else has:** Every competitor trains a model normally, then
applies SLOT at eval time. The model was never trained to BE SLOT-friendly. We train
the model WITH SLOT in the inner loop — the model learns to produce hidden states
that SLOT can adapt most effectively.

**Implementation:** Every 4th training step, instead of `loss = model(x, y)`, we:
1. Compute hidden states `h = model.forward_hidden(x)` (stays in graph)
2. Detach h, optimize a delta for 2 inner steps (FOMAML — no second-order gradients)
3. Apply the optimized delta to the ORIGINAL h (gradients flow back to model)
4. Compute and backprop the "SLOT-adapted" loss

The model learns: "given my hidden states, what delta would SLOT find, and how can I
make that delta more effective?" This shifts the entire SLOT scaling curve.

**Cost:** ~2x per meta-step (every 4th step), so ~1.25x overall training cost.
With 10 min budget, we lose ~20% of steps but gain 10-50x more SLOT effectiveness.

**Config:** META_SLOT=1, META_SLOT_EVERY=4, META_SLOT_INNER_STEPS=2, META_SLOT_INNER_LR=0.01

## Targeted Improvements (proven, small additions)

1. **Pre-quant TTT** (-0.022 BPB, from PR #1306)
2. **MuonEq-R** (-0.001 BPB, from PR #1296)
3. **QK-Gain 5.0** (vs their 4.0)
4. **WD 0.09** (vs their 0.04)
5. **L-BFGS SLOT option** (faster convergence per step)

## File Structure

```
src/train_gpt.py              — Our final submission script
src/train_gpt_pr1278_base.py  — PR #1278 reference (untouched)
```

## Decision Framework

For any change, ask:
1. Does it improve BPB by ≥0.001?
2. Does it fit in the time budget (600s train, 600s eval)?
3. Does it fit in the artifact budget (16MB)?
4. Is the code clean and the change well-understood?

If no to any → skip it.
