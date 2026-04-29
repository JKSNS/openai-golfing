# Diff against PR #1493 (bigbag)

What this submission changes from `bigbag/parameter-golf @ 857de47`. Used for
reviewer transparency and the submission PR description.

## Summary

Five new env vars, all default-off. When all are 0/unset, the runtime path is
byte-equivalent to PR #1493. The new code lives entirely in `eval_val_ttt`,
which is only called when `TTT_ENABLED=1`.

| env var | default | controls |
|---|---|---|
| `TTT_CONFIDENCE_WEIGHTED` | 0 | per-token (1 - p_correct) weighting in CE |
| `TTT_WEIGHT_FLOOR` | 0.05 | minimum per-token weight |
| `TTT_ANCHOR_LAMBDA` | 0.0 | L2 anchor regularization toward theta_0 |
| `TTT_EMA_TEACHER_LAMBDA` | 0.0 | KL distillation toward EMA-of-trajectory |
| `TTT_EMA_DECAY` | 0.99 | EMA smoothing factor |

## Source-level changes (3 changes)

### 1. Hyperparameters

Five new fields appended to the existing `ttt_*` block:

```python
ttt_confidence_weighted = bool(int(os.environ.get('TTT_CONFIDENCE_WEIGHTED', '0')))
ttt_weight_floor        = float(os.environ.get('TTT_WEIGHT_FLOOR', '0.05'))
ttt_anchor_lambda       = float(os.environ.get('TTT_ANCHOR_LAMBDA', '0.0'))
ttt_ema_teacher_lambda  = float(os.environ.get('TTT_EMA_TEACHER_LAMBDA', '0.0'))
ttt_ema_decay           = float(os.environ.get('TTT_EMA_DECAY', '0.99'))
```

### 2. Setup at top of `eval_val_ttt`

After the existing `optimizer = torch.optim.SGD(...)` line:

```python
anchor_state = [p.detach().clone() for p in ttt_params] if h.ttt_anchor_lambda > 0 else None
teacher_model = None
if h.ttt_ema_teacher_lambda > 0:
    teacher_model = copy.deepcopy(base_model)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
```

### 3. Train inner loop

The original train step:

```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = base_model(x, y)
loss.backward()
torch.nn.utils.clip_grad_norm_(ttt_params, 1.); optimizer.step()
```

becomes:

```python
if h.ttt_confidence_weighted or h.ttt_ema_teacher_lambda > 0:
    if h.ttt_confidence_weighted:
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits_eval = base_model.forward_logits(x)
            nll_eval    = F.cross_entropy(logits_eval.reshape(-1, V).float(),
                                          y.reshape(-1), reduction='none').reshape_as(y)
            cw_weights  = (1.0 - torch.exp(-nll_eval)).clamp(min=h.ttt_weight_floor)
    if h.ttt_ema_teacher_lambda > 0:
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            teacher_log_probs = F.log_softmax(teacher_model.forward_logits(x).float(), dim=-1)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits_train = base_model.forward_logits(x)
        nll_train    = F.cross_entropy(logits_train.reshape(-1, V).float(),
                                       y.reshape(-1), reduction='none').reshape_as(y)
        loss = (nll_train * cw_weights).mean() if h.ttt_confidence_weighted else nll_train.mean()
        if h.ttt_ema_teacher_lambda > 0:
            student_log_probs = F.log_softmax(logits_train.float(), dim=-1)
            kl = F.kl_div(student_log_probs.reshape(-1, V),
                          teacher_log_probs.reshape(-1, V),
                          reduction='batchmean', log_target=True)
            loss = loss + h.ttt_ema_teacher_lambda * kl
else:
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = base_model(x, y)

if h.ttt_anchor_lambda > 0:
    anchor_sq = sum(((p - a).float() ** 2).sum() for p, a in zip(ttt_params, anchor_state))
    anchor_n  = sum(p.numel() for p in ttt_params)
    loss = loss + h.ttt_anchor_lambda * anchor_sq / anchor_n

loss.backward()
# (existing all_reduce + clip_grad + step)
optimizer.step()

if h.ttt_ema_teacher_lambda > 0:
    with torch.no_grad():
        for tp, sp in zip(teacher_model.parameters(), ttt_params):
            tp.data.mul_(h.ttt_ema_decay).add_(sp.detach(), alpha=1.0 - h.ttt_ema_decay)
```

### 4. Python 3.11 compatibility

Two PEP 701 nested-quote f-strings replaced with single-quoted inner literals
(byte-identical output, just lexical fix for Python 3.11):

```python
log(f"  {cat}: {', '.join(sorted(categories[cat]))}")
log(f"train_shards: {len(list(... .glob('fineweb_train_*.bin')))}")
```

## Artifact size

| version | encoded bytes | delta |
|---|---|---|
| bigbag PR #1493 | 16,594 | - |
| this submission | 17,235 | +641 |

bigbag's full submission was 15,985,678 bytes total. +641 code bytes leaves
~14 KB of slack still under the 16,000,000 cap.

## Compliance argument

The four PR #1413 reviewer conditions hold for every combination of the
three mechanisms. See `NOVEL_ANGLE.md` for the per-mechanism walk-through.

Key points:

- All auxiliary losses (CW weights, KL teacher, anchor L2) are computed from
  the model's own state at chunk i, never from val tokens beyond chunk i.
- The `loss_sum` accumulator that produces the reported `val_bpb` is
  populated entirely in the score phase. The auxiliary losses live in the
  train phase that follows scoring; they only reshape the gradient signal,
  never the score.
- The teacher model is deep-copied from M_0 at chunk 0 and updated only via
  EMA from the student trajectory. It never sees val tokens that have not
  already been scored by the student first.
- The anchor `theta_0` is captured immediately after `deserialize`, before
  any train phase has run. It is a snapshot of the calibrated quantized model.

## Runtime impact

| config | extra forwards per step | extra eval wall |
|---|---|---|
| baseline | 0 | 0 |
| + CW | 1 (no_grad) | ~+90s |
| + CW + ART | 1 | ~+100s (ART is ~+10s) |
| + CW + ART + EMA | 2 (no_grad teacher + student) | ~+190s |

Eval budget is 600s. Bigbag eval is ~385s. Full stack at ~575s leaves ~25s
slack. ART-only adds <2% wall, EMA-Teacher is the expensive one.
