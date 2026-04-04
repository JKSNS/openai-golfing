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

### EXP-001: Baseline Reproduction
- **Date:** TBD
- **Hardware:** TBD
- **Config:** Upstream train_gpt.py (1126 lines), no modifications
- **Expected val_bpb:** ~1.2244
- **Status:** Skipped — we're starting from PR #1229 base instead

### EXP-002: PR #1229 Base (Full Standard Stack + SLOT)
- **Date:** TBD
- **Hardware:** 8×H100 SXM (RunPod)
- **Config:** PR #1229 script (2264 lines) — 11L, MLP 3×, LeakyReLU², XSA-all, EMA, BigramHash 2048×128, Full Hessian GPTQ int6, LZMA, SLOT-16
- **Expected val_bpb:** ~0.93 (with SLOT), ~1.12 (sliding only)
- **Status:** Ready to run — needs GPU access
- **Cost:** ~$3.50 per run

### EXP-003: SLOT Hyperparameter Sweep
- **Date:** TBD
- **Config:** Sweep SLOT_STEPS (8,16,24), SLOT_LR (0.005-0.012), stride (64,96)
- **Purpose:** Find optimal SLOT config within 600s eval budget
- **Status:** Planned

### EXP-004: BigramHash Scaling
- **Date:** TBD
- **Config:** BigramHash 2048→3072 vocab, dim 128→112
- **Purpose:** Match SOTA's bigram config within artifact budget
- **Status:** Planned

### EXP-005: Pre-Quant TTT
- **Date:** TBD
- **Config:** Add pre-quant TTT (PR #1306 approach) before GPTQ
- **Purpose:** -0.022 BPB from TTT before quantization
- **Status:** Planned
