# Strategy

## Where the leaderboard is (2026-04-26)

| Rank | Score | PR | Recipe |
|---|---|---|---|
| 1 | 1.0810 | #1493 | bigbag: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT |
| 2 | 1.0822 | #1477 | aryanbhosale: SP8192 + Parallel Residuals + Score-First TTT |
| 3 | 1.0828 | #1413 | dexhunter: SP8192 + QK-Gain 5 + Legal Score-First TTT |
| 4 | 1.0835 | #1412 | Robby Sneiderman: Parallel Residuals + Hessian-Aware SDClip |
| 5 | 1.0856 | #1394 | Kevin Clark: SP8192 + GPTQ Embeddings + Depth Recurrence + SDClip (the base) |

PR #1394 (Kevin Clark) is the shared base. Everyone above iterates on it.
Deadline: 2026-04-30. Today: 2026-04-26. Four days.

## What changed under me

Old plan: SLOT (eval-time delta optimization) + mean-delta warm-start. That lineage
is dead. Issue #1336 left SLOT legality unresolved and the community converged on
**Legal Score-First TTT** instead (PR #461 origin). Score-first TTT trains the
actual model on each val chunk *after* scoring it under inference_mode. The score
is locked in before any update touches the chunk. That is the only TTT pattern in
the top 5.

My SLOT mean-delta warm-start does not port to score-first TTT (no shared external
delta to carry). I am abandoning the SLOT branch.

## What I am building on

Base: PR #1493 by bigbag at 1.0810 BPB. Self-contained submission at
`bigbag/parameter-golf @ 857de47` under
`records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/`.
The train_gpt.py is lzma+base85 self-extracting (16,594 bytes). Decoded source is
469 logical lines.

Stack in PR #1493:
- SP8192 vocab, 11 layers, 512 dim, MLP 4x. 11 physical + 2 extra passes over L3-L5
  = 17 virtual layers.
- Parallel residuals from L7+
- QK-Gain 5.25 (per-head learnable Q scaling)
- Legal Score-First TTT (TTT_LR=0.005, TTT_EPOCHS=3, TTT_CHUNK_TOKENS=32768,
  freeze_blocks=0)
- GPTQ embeddings (8 bit, clip_sigmas=20.0), MATRIX_BITS=6, brotli compression
- ~4550 training steps in 600s, ~385s eval

Artifact size: 15.99 MB (under 16,000,000 decimal cap by ~14 KB).

## The novel angle: Confidence-Weighted Score-First TTT

Currently the train phase that follows scoring chunk i uses **uniform** per-token
cross-entropy. The model already has per-token confidence info from the score-first
pass we just paid for. We are throwing it away.

Replace plain CE with `(1 - p_correct).clamp(min=floor)`-weighted CE during the
train phase. Scoring is unchanged; only the gradient signal during the SGD updates
is reshaped to focus on tokens the model is uncertain about.

Why this is novel:
- Nobody in the top 5 does it. They all use uniform CE in the train phase.
- Compliant: the score is locked in BEFORE any weight is computed. Per-token
  weights only affect the gradient on val data we are already legally training on
  (PR #461 score-first pattern).
- Falsifiable: one $7 run measures the delta on top of PR #1493.
- Clean: a single multiplicative weight in the loss, no architecture change.
- Stacks with TTT_LR / TTT_EPOCHS sweeps (orthogonal axes).

Expected upside is modest (~0.001-0.003 nats). Probably not enough alone to beat
1.0810 by 0.005 at p<0.01, but combined with hyperparameter sweeps it has a real
shot.

Full writeup, including the four-condition compliance check against PR #1413
reviewer language, lives in `NOVEL_ANGLE.md`. The patched submission lives in
`patches/`.

## The four-day plan

| Phase | Cost | What |
|---|---|---|
| 0. Reproduce PR #1493 | $7 | Single seed of bigbag's recipe. Validator must pass all gates. |
| 1. CW-TTT smoke | $7 | TTT_CONFIDENCE_WEIGHTED=1, TTT_WEIGHT_FLOOR=0.05, single seed. |
| 2. Decision gate | - | If CW-TTT >=0.001 better, sweep floor. Else fall back to clean repro. |
| 3a. Floor sweep | $28 | floor in {0, 0.05, 0.1, 0.25} single seed each. |
| 3b. TTT_LR sweep | $14 | lr in {0.003, 0.007} on the best floor. |
| 4. 3-seed validate | $21 | Best config across seeds 42, 314, 999. Aggregate.py confirms p<0.01. |
| 5. Submit | - | Either record (mean <= 1.0760 with bound) or non-record validation of bigbag's recipe. |

Total: $77. About 8% of the $996 grant. Plenty of headroom for follow-ups if a
sweep reveals an unexpected win.

## Compliance feedback from PR #1368

The community review (@MatoTeziTanka) flagged my old PR for val-leak in pre-quant
TTT (`ttt_adapt_adamw` running 6 epochs of train-on-val with no score-first
discipline). The fix is to use score-first chunked TTT (which is what the top
entries do) or pre-quant TTT on a held-out training slice (PR #1416/#1423).

I am taking the score-first path because:
1. It is what is winning.
2. It is what bigbag's PR #1493 already does correctly.
3. My novel angle is layered ON TOP of score-first, never around it.

The four-condition checklist from PR #1413 reviewer language:
1. Causality (TTT only sees tokens at or before the position being scored)
2. Normalized softmax (not raw logits)
3. Score-before-update (score under inference_mode, then train)
4. Single-pass (no rescoring after training)

PR #1493's TTT block satisfies all four. My CW-TTT patch preserves all four
because the per-token weights are derived from the model's predictions on val data
the train phase is already legally going to train on. No new val data is touched.
The val_bpb logged at the end is byte-for-byte the same as if I had not used the
weights, except the model state has been driven by a tighter gradient.
