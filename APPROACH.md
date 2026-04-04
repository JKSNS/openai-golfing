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

## Novel Technique (TBD — researching)

Candidates under investigation:
- **SLOT-Aware Meta-Training**: Train model to be maximally SLOT-improvable
- **Multi-Resolution SLOT**: Deltas at multiple layers
- **Amortized SLOT Init**: Learned delta predictor
- **Online Bayesian Mixture**: Replace SLOT with theoretically-grounded approach

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
