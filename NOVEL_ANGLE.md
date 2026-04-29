# Stabilized Score-First TTT

The novel research direction. Three composable stability mechanisms layered on
top of the score-first TTT pattern (PR #461 origin, refined in PR #1413/#1493).
All three address the same underlying problem - TTT updates accumulate noise
across chunks, degrading the model that scores late chunks - via three
different mechanisms.

## The problem

In score-first TTT (the only legal TTT pattern in the top 5 of the merged
leaderboard):

```
chunk 0: score with M_0, train -> M_1
chunk 1: score with M_1, train -> M_2
...
chunk N-1: score with M_{N-1}, train (skipped on last chunk)
```

Late chunks score against a model that has been trained on all earlier
chunks. The compounding gain from training is the entire point of TTT - but
it cuts both ways. If individual-chunk gradients are noisy, that noise
propagates and the model that scores chunk N-1 is *more polluted* than the
one that scored chunk 0.

The cosine LR schedule in PR #1493 (`cos_lr = ttt_lr * 0.5 * (1 + cos(pi * ci /
N))`) goes high-to-low, partially countering this by giving late chunks
gentler updates. But the actual *parameter trajectory* is unconstrained: by
chunk N-1, the model has wandered an unbounded distance from M_0.

## Three stability mechanisms

### 1. Confidence-Weighted TTT (CW-TTT)

```python
weights = (1.0 - exp(-nll_eval)).clamp(min=ttt_weight_floor)   # under no_grad
loss = (nll_train * weights).mean()
```

Reshapes *which tokens* drive the gradient. Tokens the model is already
confident about contribute weight `floor`; tokens it is uncertain about
contribute up to weight 1. The information content of the gradient signal
shifts toward the tokens where there is something to learn.

Compliance: the per-token weights are computed under `torch.no_grad()` from
the model's predictions on the chunk the train phase is already legally
training on. No new validation tokens are touched. The score for that chunk
is locked in before the train phase begins.

Cost: one extra inference-mode forward per train step. ~+90s eval wall.

### 2. Anchor-Regularized TTT (ART)

```python
anchor_state = [p.detach().clone() for p in ttt_params]   # snapshot at chunk 0
# ... in the train loss ...
anchor_sq = sum(((p - a).float() ** 2).sum() for p, a in zip(ttt_params, anchor_state))
loss = loss + ttt_anchor_lambda * anchor_sq / num_params
```

Adds an L2 penalty pulling theta back toward theta_0 (the deserialized
quantized weights at the start of TTT). The model is allowed to drift, but
the drift is bounded.

Why this should help: the deserialized model has been calibrated for the val
distribution by training; TTT updates can only refine that, but unbounded
drift moves the model out of the calibrated region. ART says "you can adapt,
but stay close to the calibrated point."

Cost: one elementwise diff and sum per train step over ~30M params. ~+10s
eval wall. Negligible.

Hyperparameter: `ttt_anchor_lambda`. Sweep candidates: `{0.1, 1.0, 10.0}`.

### 3. EMA-Teacher TTT

```python
teacher_model = copy.deepcopy(base_model)   # at chunk 0
# ... in the train loss ...
with torch.no_grad():
    teacher_log_probs = log_softmax(teacher_model.forward_logits(x))
student_log_probs = log_softmax(logits_train)
kl = F.kl_div(student_log_probs, teacher_log_probs, reduction='batchmean', log_target=True)
loss = loss + ttt_ema_teacher_lambda * kl
# ... after optimizer.step() ...
for tp, sp in zip(teacher_model.parameters(), ttt_params):
    tp.data.mul_(ema_decay).add_(sp.detach(), alpha=1 - ema_decay)
```

Maintains an EMA copy of the model trajectory. Adds `KL(student || teacher)`
as auxiliary loss. The student is pulled toward agreement with the
slowly-evolving EMA, which averages out per-chunk gradient noise.

Why this should help: classic noise-stabilization trick from semi-supervised
learning (mean teacher / temporal ensembling). Each individual chunk's
gradient has noise; averaging across chunks via EMA gives a stable target.

Cost: one extra forward per train step (teacher, no_grad). One deepcopy of
the model at chunk 0 (~120 MB GPU memory for the float32 teacher state).
~+90s eval wall.

Hyperparameters: `ttt_ema_teacher_lambda`, `ttt_ema_decay`. Defaults
`0.5` and `0.99`.

## Why this is genuinely novel

Searched the leaderboard top entries (PRs #1493, #1477, #1413, #1412, #1394)
plus the unmerged claims (PR #1911, #1908, #1915). None of them do per-token
weighting, anchor regularization, or EMA-teacher distillation in the
score-first TTT train phase. The TTT lineage uses vanilla SGD with momentum
0.9 and uniform CE.

The closest prior art:

- **Mean-teacher / temporal ensembling** (Tarvainen & Valpola 2017) for
  semi-supervised learning - same EMA mechanism but applied to labeled+unlabeled
  training, not eval-time TTT.
- **Elastic weight consolidation** (Kirkpatrick et al. 2017) for continual
  learning - same anchor mechanism but with Fisher information weighting,
  applied across tasks not chunks.
- **Focal loss** (Lin et al. 2017) - same per-token reweighting but for
  imbalanced object detection.

This is the first application I can find of any of these mechanisms to
score-first TTT for language model evaluation in a parameter-constrained
challenge.

## Compliance walk-through (PR #1413 four conditions)

For all three mechanisms:

1. **Causality.** All weights/teachers/anchors are derived from the model's
   own state and its predictions on tokens the train phase is already legally
   training on. No future val tokens leak. Anchor uses a snapshot of M_0,
   which existed before any val token was scored.
2. **Normalized softmax.** All forwards use the same `forward_logits` ->
   `cross_entropy` chain or `log_softmax`. No raw logit shenanigans.
3. **Score-before-update.** The `loss_sum` accumulator that drives val_bpb
   is populated entirely in the score phase, before any of these auxiliary
   losses are computed. The train phase is what gets reshaped.
4. **Single-pass.** Each chunk is scored exactly once. The auxiliary losses
   only affect the gradient signal during training, never trigger a re-score.

## Test plan

| run | seeds | floor | anchor | ema | cost |
|---|---|---|---|---|---|
| Phase 0 (baseline) | 42 | - | - | - | $7 |
| Phase 1 (CW-TTT) | 42 | 0.05 | - | - | $7 |
| Phase 2 (CW + ART) | 42 | 0.05 | 1.0 | - | $7 |
| Phase 2b (CW + ART + EMA) | 42 | 0.05 | 1.0 | 0.5 | $8 |
| Sweep top 1 (best mech) | 42 | sweep | sweep | sweep | $30 |
| 3-seed validate winner | 42, 314, 999 | best | best | best | $24 |

Total ~$83. Well within $1000 grant. Runnable in 1 day.

## Decision tree

After Phase 2 and Phase 2b complete:

- If both regress vs Phase 1: CW-TTT alone is the submission.
- If ART helps (+CW < +CW+ART): drop EMA-Teacher, sweep anchor_lambda.
- If EMA-Teacher helps and ART doesn't: drop ART, sweep ema_lambda/decay.
- If both help additively: full Stabilized TTT stack is the submission.
- If they help individually but interact badly: pick the better single one.

The 3-seed validation tells us if the chosen mechanism is real signal or
single-seed noise.

## Submission framing

> "I introduce three orthogonal stability mechanisms for score-first TTT:
> Confidence-Weighted CE, Anchor regularization, and EMA-Teacher
> distillation. Each addresses TTT trajectory noise via a different
> mechanism. The full Stabilized TTT stack achieves val_bpb X.XXXX +/- Y.YYYY
> across 3 seeds, beating PR #1493 by Z.ZZZ at p<0.01. All three mechanisms
> preserve the four PR #1413 reviewer conditions and operate exclusively in
> the score-first compliance regime."
