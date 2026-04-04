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
- **Config:** Upstream train_gpt.py, no modifications
- **Expected val_bpb:** ~1.2244
- **Status:** Not started
