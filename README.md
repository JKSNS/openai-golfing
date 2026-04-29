# openai-golfing

Going after first place on OpenAI's Parameter Golf challenge. Current SOTA: 1.0810 BPB
(bigbag PR #1493). Deadline: 2026-04-30.

## Two commands

```bash
cd /workspace && git clone https://github.com/JKSNS/openai-golfing.git
bash openai-golfing/scripts/run_all.sh
```

That runs the full Stabilized TTT ablation series end-to-end: setup → preflight →
Phase 0 (baseline) → Phase 1 (+ CW-TTT) → Phase 2 (+ ART) → Phase 2b (+ EMA-Teacher) →
compare → decision banner.

Total: **~80 min wallclock, ~$29 compute**. Use the official Parameter Golf RunPod
template for an 8xH100 SXM pod.

## What this submission introduces

Three composable stability mechanisms for score-first TTT, all novel for this
challenge:

| layer | env var | mechanism |
|---|---|---|
| **CW-TTT** | `TTT_CONFIDENCE_WEIGHTED=1`, `TTT_WEIGHT_FLOOR=0.05` | per-token (1 - p_correct) gradient reshaping |
| **ART** | `TTT_ANCHOR_LAMBDA=1.0` | L2 anchor pulling theta toward theta_0 |
| **EMA-Teacher** | `TTT_EMA_TEACHER_LAMBDA=0.5`, `TTT_EMA_DECAY=0.99` | KL distill toward EMA-of-trajectory |

All three address the same root problem (TTT trajectory noise propagating to late
chunks via different mechanisms) and all preserve the four PR #1413 reviewer
conditions. See `NOVEL_ANGLE.md` for the full writeup, `DIFF.md` for the source diff.

## Skip flags (for partial reruns)

```bash
SKIP_SETUP=1     SKIP_PREFLIGHT=1     # skip env work (already done)
SKIP_P0=1 SKIP_P1=1 SKIP_P2=1 SKIP_P2B=1   # skip individual phases
```

## After the table prints

`compare.py` and the decision banner tell you which phase won. Then:

```bash
# Validate the winning config across 3 seeds (~$24, ~60 min)
bash scripts/run_3seed.sh cwttt 0.05         # if Phase 1 won
bash scripts/run_3seed.sh cwttt 0.05         # ART/EMA stack: edit args inline
bash scripts/run_3seed.sh repro              # if nothing beat baseline

# Build a records/ folder ready for PR
bash scripts/package_submission.sh runs/3seed_<config>_<ts>
```

## Pod teardown

```bash
runpodctl stop pod <pod-id>     # ~$0.01/hr storage when stopped
```

## File map

- `STRATEGY.md` - leaderboard map, why building on PR #1493, novel stack rationale.
- `NOVEL_ANGLE.md` - full Stabilized TTT writeup with compliance argument.
- `DIFF.md` - source diff against bigbag PR #1493.
- `RUNBOOK.md` - long-form runbook (this README is the short version).
- `scripts/run_all.sh` - the single entrypoint.
- `scripts/setup.sh` - idempotent pod setup.
- `scripts/preflight.sh` - 5-second env check.
- `scripts/smoke.sh` - 60-step micro-train (~$0.20 sanity check).
- `scripts/run_phase0.sh` - bigbag baseline reproduction.
- `scripts/run_phase1.sh` - + CW-TTT.
- `scripts/run_phase2.sh` - + CW + ART.
- `scripts/run_phase2b.sh` - full Stabilized TTT stack.
- `scripts/run_phase3a.sh` - floor sweep over CW-TTT.
- `scripts/run_3seed.sh` - 3-seed validation runner.
- `scripts/validate.py` - per-log gate check (time, score range, compliance, size).
- `scripts/compare.py` - Welch t-test, Cohen's d, 99% CI on deltas.
- `scripts/aggregate.py` - mean/std + p<0.01 record-margin check.
- `scripts/report.py` - per-run JSON + markdown report with metadata.
- `scripts/monitor.sh` - live progress dashboard for a running log.
- `scripts/package_submission.sh` - build records/ folder for the upstream PR.
- `patches/train_gpt_cwttt.py` - readable source with all three mechanisms.
- `patches/train_gpt.py` - re-encoded self-extracting submission (17,235 bytes,
  +641 vs bigbag).
- `patches/encode_submission.py` - re-encode helper for future edits.

## Compliance

PR #1368 was flagged 2026-04-14 for val-leak in pre-quant TTT. The path forward is
score-first chunked TTT (PR #461 origin, refined in PR #1413). PR #1493 already
implements this correctly; my three new mechanisms layer on top without violating
any of the four reviewer conditions. See `NOVEL_ANGLE.md` for the per-mechanism
walk-through.
