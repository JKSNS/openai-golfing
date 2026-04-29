# Runbook

The well-oiled flow. Pod -> reproduce -> CW-TTT smoke -> decide -> sweep -> validate -> submit.

## Cost reality

8xH100 SXM on RunPod: ~$20-25/hr running. A stopped pod is ~$0.01/hr storage
only (this is what tripped me up).

Budget: $996.66 grant. ~40 hours running. ~240 ten-minute runs.

## Pod setup (5 min)

1. Deploy 8xH100 SXM via the official Parameter Golf RunPod template (Hopper
   required for Flash-Attention 3). Enable SSH terminal.

2. SSH in. Run the bootstrap:

   ```bash
   cd /workspace
   curl -fsSL https://raw.githubusercontent.com/JKSNS/openai-golfing/main/scripts/bootstrap.sh | bash
   ```

   This clones bigbag's fork at commit 857de47 (the PR #1493 head), prefetches
   FineWeb SP8192, clones this repo for the validator and patches, and verifies
   CUDA + FA3.

3. Copy the patched submission into the bigbag fork:

   ```bash
   cp /workspace/openai-golfing/patches/train_gpt.py \
      /workspace/parameter-golf/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt_cwttt.py
   ```

   Note: the original bigbag train_gpt.py stays untouched. The CW-TTT variant
   lives alongside it. Run whichever one you want by changing the script name.

## Phase 0: Reproduce PR #1493 (single seed, $7)

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT
SEED=42 \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 \
  | tee /workspace/logs/repro_seed42.log
python3 /workspace/openai-golfing/scripts/validate.py /workspace/logs/repro_seed42.log
```

Expected `quantized_ttt val_bpb` line: 1.0810 +/- 0.0010. If it lands outside
[1.078, 1.084], something is wrong with the pod (FA3 missing, dataset version
mismatch, etc).

## Phase 1: Confidence-Weighted TTT smoke (single seed, $7)

```bash
SEED=42 \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
TTT_CONFIDENCE_WEIGHTED=1 \
TTT_WEIGHT_FLOOR=0.05 \
torchrun --standalone --nproc_per_node=8 train_gpt_cwttt.py 2>&1 \
  | tee /workspace/logs/cwttt_seed42.log
python3 /workspace/openai-golfing/scripts/validate.py /workspace/logs/cwttt_seed42.log
```

Compare the `quantized_ttt val_bpb` line to Phase 0. Decision gate:

- delta <= -0.001: keep going. Sweep floor in Phase 3a.
- -0.001 < delta < +0.001: noise. Run a second seed before deciding.
- delta >= +0.001: drop CW-TTT. Submit the clean reproduction.

The eval should be ~475s instead of ~385s due to the extra forward pass per train
step. Still well under the 600s eval cap.

## Phase 3a: Floor sweep (4 runs, $28) - only if Phase 1 looks good

```bash
for FLOOR in 0.0 0.05 0.10 0.25; do
  SEED=42 \
  QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  TTT_CONFIDENCE_WEIGHTED=1 TTT_WEIGHT_FLOOR=$FLOOR \
  torchrun --standalone --nproc_per_node=8 train_gpt_cwttt.py 2>&1 \
    | tee /workspace/logs/cwttt_floor${FLOOR}_seed42.log
done
python3 /workspace/openai-golfing/scripts/validate.py /workspace/logs/cwttt_floor*_seed42.log
```

Pick the best floor. If multiple are within 0.0005, prefer the one closest to
0.05 (most conservative).

## Phase 3b: TTT_LR sweep on the winner (2 runs, $14)

```bash
BEST_FLOOR=0.05  # from Phase 3a
for LR in 0.003 0.007; do
  SEED=42 \
  QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=$LR TTT_EPOCHS=3 \
  TTT_CONFIDENCE_WEIGHTED=1 TTT_WEIGHT_FLOOR=$BEST_FLOOR \
  torchrun --standalone --nproc_per_node=8 train_gpt_cwttt.py 2>&1 \
    | tee /workspace/logs/cwttt_lr${LR}_seed42.log
done
```

## Phase 4: 3-seed validate the winning config ($21)

```bash
WINNING_LR=0.005       # from Phase 3b
WINNING_FLOOR=0.05     # from Phase 3a
for SEED in 42 314 999; do
  SEED=$SEED \
  QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=$WINNING_LR TTT_EPOCHS=3 \
  TTT_CONFIDENCE_WEIGHTED=1 TTT_WEIGHT_FLOOR=$WINNING_FLOOR \
  torchrun --standalone --nproc_per_node=8 train_gpt_cwttt.py 2>&1 \
    | tee /workspace/logs/cwttt_final_seed${SEED}.log
done
python3 /workspace/openai-golfing/scripts/aggregate.py /workspace/logs/cwttt_final_seed*.log
```

The aggregator reports mean +/- std and the p<0.01 confidence margin. To clear
the record bar, the upper bound of the 99% CI must be below 1.0760 (PR #1493's
1.0810 minus 0.005 nats).

## Submission gate

Before opening a PR:

1. 3-seed mean beats 1.0810 by >=0.005 at p<0.01 (aggregate.py says RECORD).
2. validate.py passes on every seed log (time budgets, score range, score-first
   compliance marker, artifact size).
3. Diff against PR #1493 is minimal (only the eval_val_ttt train inner loop
   plus two new env vars).
4. submission.json filled with my GitHub handle, val_bpb mean, std, seed list.
5. README.md describes the angle and references PR #1413's four-condition
   compliance checklist.

If 1 fails but the validator passes: submit as a non-record validation of
PR #1493. Frame honestly. Better than nothing, leaves room for follow-up.

## Teardown

```bash
# Pull logs back to local
scp -r runpod:/workspace/logs ~/openai-golfing-runs/$(date +%Y%m%d)
# Stop the pod (do NOT delete it; storage is cheap, recreation is not)
runpodctl stop pod <pod-id>
```
