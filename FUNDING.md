# Parameter Golf — Compute Grant Application

## Grant Info

- **Form:** https://openai.com/index/parameter-golf/#credit-form
- **Pool:** $1,000,000 total (distributed until depleted)
- **Provider:** RunPod credits
- **Requirement:** Submit with email tied to OpenAI/ChatGPT account
- **Turnaround:** ~5 business days
- **We currently have:** $25 in RunPod credits (~7 full submission runs)

## Cost Reference

| Config | $/hr | Notes |
|--------|------|-------|
| 1×H100 SXM | ~$2.69 | Development/iteration |
| 8×H100 SXM | ~$20–22 | Submission runs |
| 1×RTX 4090 | ~$0.50 | Budget prototyping |

**Per submission run** (10 min on 8×H100): **~$3.50**

| Credits | 8×H100 Hours | Full Runs | 1×H100 Hours |
|---------|-------------|-----------|---------------|
| $25 | 1.2 hrs | ~7 | ~9 hrs |
| $100 | 4.8 hrs | ~28 | ~37 hrs |
| $200 | 9.5 hrs | ~57 | ~74 hrs |
| $500 | 24 hrs | ~143 | ~186 hrs |

## Draft Application

**Requested Level:** [Select appropriate tier in form — target $100–200]

### Technical Plan

**Phase 1 — Local Prototyping (Days 1–3, $0)**
- Reproduce baseline architecture and understand codebase
- Implement and test core modifications locally (MLX or CPU)
- Validate approach shows signs of life before spending compute

**Phase 2 — Cheap GPU Iteration (Days 4–10, ~$30–50 on 1×H100)**
- Port modifications to CUDA, iterate on hyperparameters
- Key experiments:
  - Int6 QAT + zstd-22 compression vs baseline int8+zlib
  - 11L + MLP 3× architecture scaling
  - XSA (exclusive self-attention) on all layers
  - LeakyReLU² activation + EMA weight averaging
- Decision gate: proceed to Phase 3 only if beating baseline by >0.02 BPB

**Phase 3 — 8×H100 Validation (Days 11–20, ~$40–80)**
- Scale winning configurations to 8×H100
- 10–20 full submission runs to measure variance
- Tune final hyperparameters (warmdown, LR schedule, quantization timing)
- Implement and test GPTQ with self-generated calibration

**Phase 4 — Final Submission (Days 21–27, ~$20–40)**
- 5–10 final runs for statistical significance (p < 0.01)
- Prepare PR with logs, README, submission.json
- Document all findings including negative results

**Total estimated spend: $100–200**

### Why This Is Worth Funding

- Phased approach with explicit go/no-go gates prevents waste
- Majority of iteration on 1×H100 (~$2.69/hr), not 8×H100 (~$20/hr)
- Will contribute findings via PR regardless of final ranking
- Exploring [specific novel technique] in addition to proven stack
- 27 days remaining — tight but achievable timeline

### Efficient Compute Usage

- Local prototyping for zero-cost validation
- 1×H100 for 80% of experiments
- 8×H100 only for final validation and submission runs
- Each run is ~$3.50 — budget covers sufficient statistical replication

### Background

[Fill in: relevant ML/systems experience, competition history, publications]

## Tips for Strong Application

1. Pick the lowest credit level that covers your plan
2. Reference the challenge's "Requests for PRs" wishlist if applicable
3. Mention any work already done (local experiments, baseline reproduction)
4. Emphasize phased approach — shows compute literacy
5. Commit to sharing results (even negative) as PRs
