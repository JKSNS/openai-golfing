# Experiment Log

Track every experiment with config, result, and takeaway.

## Template

```
### EXP-NNN: [Name]
- **Date:** YYYY-MM-DD
- **Hardware:** 1×H100 / 8×H100
- **Config changes:** [what changed from previous best]
- **Steps:** N
- **val_bpb:** X.XXXX (mean ± std over N seeds)
- **Artifact size:** XX.XX MB
- **Step time:** XX.X ms
- **Takeaway:** [what we learned]
```

---

## Experiments

### EXP-001: Smoke Test (v0.3 full stack)
- **Date:** TBD
- **Hardware:** 1×H100 (or any GPU)
- **Config:** Full v0.3 stack, 50 iterations, SLOT_STEPS=0 (no SLOT for speed)
- **Purpose:** Verify script runs, loss decreases, artifact < 16MB
- **Command:** `bash scripts/smoke_test.sh`
- **Cost:** ~$0.10
- **Status:** Not started

### EXP-002: Full Run — L-BFGS SLOT (v0.3)
- **Date:** TBD
- **Hardware:** 8×H100 SXM (RunPod)
- **Config:** Full v0.3 stack — 11L, depth recurrence, MuonEq-R, TTT, L-BFGS SLOT-10
- **Expected val_bpb:** ~0.72-0.76 (with SLOT), ~1.07-1.09 (sliding only)
- **Command:** `bash scripts/full_submission.sh 1337`
- **Cost:** ~$3.50
- **Status:** Ready to run

### EXP-003: AdamW SLOT-24 (v0.2 config for comparison)
- **Date:** TBD
- **Config:** Same base, but SLOT_LBFGS=0, SLOT_STEPS=24, SLOT_LR=0.012
- **Purpose:** Compare AdamW vs L-BFGS SLOT quality
- **Status:** Planned

### EXP-004: SP4096 + MLP 4× Variant
- **Date:** TBD
- **Config:** VOCAB_SIZE=4096, MLP_MULT=4.0, rest same as v0.3
- **Purpose:** Test larger vocabulary path (PR #1296 achieves 1.0897 sliding-only)
- **Setup:** `bash scripts/setup_sp4096.sh`
- **Risk:** Larger embedding may require more aggressive pruning
- **Status:** Planned

### EXP-005: SLOT Timing Sweep
- **Date:** TBD
- **Config:** Vary SLOT_LBFGS_MAX_ITER (5,8,10,15,25), measure time + BPB
- **Purpose:** Find optimal iter count within 600s eval budget
- **Status:** Planned

### EXP-006: TTT Ablation
- **Date:** TBD
- **Config:** TTT_ENABLED=0 vs TTT_ENABLED=1
- **Purpose:** Measure actual TTT impact on our stack (may not help like PR #756 found)
- **Status:** Planned

### EXP-007: 3-Seed Validation (final)
- **Date:** TBD
- **Hardware:** 8×H100 SXM
- **Config:** Best config from EXP-002/003/005
- **Command:** `bash scripts/three_seed_validation.sh`
- **Cost:** ~$10.50
- **Purpose:** Statistical significance for submission (p < 0.01)
- **Status:** Blocked on EXP-002 results
