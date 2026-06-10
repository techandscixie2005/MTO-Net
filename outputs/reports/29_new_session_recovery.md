# New Session Recovery Report — MTO-Net

**Report**: 29_new_session_recovery.md
**Date**: 2026-06-10
**Session**: Fresh Ralph recovery
**Branch**: tensor-mto-v2

---

## 1. Repository State

- **Local path**: `/home/xiangyu_xie/MTO`
- **HPC path**: `/data/home/scwc008/run/xxy/MTO` (resolves to `/data/run01/scwc008/xxy/MTO`)
- **Remote**: `git@github.com:techandscixie2005/MTO-Net.git`
- **Branch**: `tensor-mto-v2`
- **Current commit**: `2b4cbd5` — "chore: remove unused imports in figure generation script"
- **Dirty tree**: 4 modified figures + 19 untracked files (scripts, jobs, .omc/, third_party examples)

## 2. Job Status (Verified 2026-06-10)

| Job ID | Name | sacct State | Actual Outcome |
|--------|------|-------------|----------------|
| 86187 | mto_transfer | COMPLETED (0:0) | COMPLETED — but raw alpha (~23.6), subspace sim all null |
| 86189 | mto_ablation | COMPLETED (0:0) | COMPLETED — but only 1 epoch, raw metrics |
| 86186 | mto_frozen | FAILED (1:0) | FAILED — RuntimeError: size mismatch (hidden_dim 64 vs 32) |
| 86198 | mto_frozen | COMPLETED (0:0) | FAILED — same size mismatch, job ran 22s then crashed |

## 3. Experiment Results Assessment

### Stage Transfer (86187)
- **Output**: `outputs/stage_transfer/stage_transfer_results.json` (15KB, 3 seeds × 3 conditions)
- **Issues**:
  - alpha MAE values are RAW (~23.6) — need normalization
  - mu MAE values appear RAW (~1.05) — need normalization
  - Subspace similarity (S_sub) is `null` for all pairs — not computed
  - Only 1 epoch trained per condition (was supposed to be 20)
- **Fix needed**: Add normalization to metrics, compute subspace similarity, train more epochs

### Full Ablation (86189)
- **Output**: `outputs/metrics/baselines/baseline_comparison.json` (seed2 only saved)
- **Issues**:
  - Only **1 epoch** trained per method (should be 20)
  - All metrics are RAW (alpha ~23.6, mu ~1.05)
  - Metrics JSON only contains seed2 results (last seed overwrites)
- **Fix needed**: Fix epochs parameter, add normalized metrics, save per-seed results

### Frozen Probe (86198)
- **Output**: NONE — no `outputs/frozen_probe/` directory exists
- **Root cause**: `size mismatch` — checkpoint has hidden_dim=64 but script instantiates model with `mto_hidden_dim=32`
- **Fix needed**: Check actual Stage A checkpoint hidden_dim, match in frozen probe script

## 4. Existing Reports (verified with real content)

All 44 report files in `outputs/reports/` contain real content. Key reports:
- 12_final_artifact_audit.md — comprehensive audit
- 13_missing_evidence_plan.md — experiment plan
- 14_frozen_probe.md — preliminary (placeholder results)
- 15_stage_transfer.md — preliminary (partial results)
- 16_full_ablation.md — preliminary (placeholder results)
- 17_stage_b_label_limitation.md — complete
- 18_stage_c_label_limitation.md — complete
- 19_final_figure_audit.md — complete
- 20_final_story.md — complete
- 21_claim_evidence_table.md — complete (well-structured)
- 22_limitations.md — complete
- 23_data_and_code_availability.md — complete
- 24_nature_reviewer_self_critique.md — complete
- 25_codex_review.md — framework only (actual review not run)
- 26_codex_review_response.md — placeholder
- 27_final_verification.md — partial (pending experiments)

## 5. Existing Figures

| # | Figure | Status |
|---|--------|--------|
| 1-15 | Stage A figures | FINAL — real data |
| 16 | Baseline ablation | PRELIMINARY — smoke-scale, needs regen |
| 17 | Stage transfer | PRELIMINARY — placeholder, needs regen |
| 18 | Frozen probe | PRELIMINARY — placeholder, needs regen |
| S | Stage B spectral smoke | PRELIMINARY — smoke only |

## 6. Test Status

```
75 passed, 19 skipped in 11.98s (verified locally)
```

## 7. Dataset Labels

| Label | Available | Location |
|-------|-----------|----------|
| mu (dipole) | Yes | qm9s.pt |
| alpha (polarizability) | Yes | qm9s.pt |
| IR spectrum CSV | Yes (8.2 GB) | HPC only |
| Raman spectrum CSV | Yes (8.2 GB) | HPC only |
| UV spectrum CSV | Yes (1.5 GB) | HPC only |
| Hessian/frequencies/normal modes | No | Not in QM9S |

## 8. What Remains

1. Fix frozen probe (dimension mismatch) → re-submit
2. Fix ablation (epochs + normalization) → re-submit
3. Fix stage transfer (normalization + subspace sim) → re-run analysis
4. Wait for jobs to complete
5. Regenerate figures 16-18
6. Metric normalization audit + fix
7. Codex review
8. Final verification
9. Commit and push
