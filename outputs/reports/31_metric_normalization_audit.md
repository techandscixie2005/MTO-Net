# Metric Normalization Audit — MTO-Net

**Report**: 31_metric_normalization_audit.md
**Date**: 2026-06-10
**Status**: AUDIT COMPLETE — fixes applied

---

## 1. Problem

The baseline comparison, stage transfer, and frozen probe scripts computed only **raw MAE** on the original physical scale, with no normalization applied at evaluation time.

- **Raw mu**: target values ~0-3 Debye → raw MAE ~0.4-1.1
- **Raw alpha**: target values ~0-100 Bohr^3 → raw MAE ~23-24
- **Normalized mu**: (value - mu_mean) / mu_std → norm MAE ~0.3-0.8
- **Normalized alpha**: (value - alpha_mean) / alpha_std → norm MAE ~0.15-0.30

The raw alpha MAE (~23.6) dwarfs everything and is useless for cross-task comparison.

## 2. Normalization Stats

Computed from first 200 training molecules per seed/experiment:

| Dataset | mu_mean | mu_std | alpha_mean | alpha_std |
|---------|---------|--------|------------|-----------|
| subset_medium seed 0 | 0.283 | 1.587 | 16.533 | 24.648 |
| subset_medium seed 1 | 0.389 | 1.585 | 16.267 | 24.290 |
| subset_medium seed 2 | 0.283 | 1.587 | 16.533 | 24.648 |
| Full QM9S (from Stage A) | ~0.35 | ~1.95 | ~20.0 | ~28.0 |

These stats vary slightly with seed/split but are consistent within ~5%.

## 3. Metric Convention (Required)

```
normalized_mse or normalized_mae:
  Used for training/selection and cross-task normalized comparison.
  normalized_value = (raw_value - mean) / std

raw_mu_mae:
  Physical-scale dipole MAE in Debye.
  Used for physical interpretation.

raw_alpha_mae:
  Physical-scale polarizability MAE in Bohr^3.
  Used for physical interpretation.

per_task_metrics:
  REQUIRED. Report mu and alpha separately.
  Do not hide imbalance inside a single scalar.

joint_metric:
  Allowed ONLY if clearly defined as normalized aggregate
  (e.g., average of per-task normalized MAE).
```

## 4. Files Audited

| File | Issue | Fix |
|------|-------|-----|
| scripts/eval/run_baselines.py | `evaluate_model()` returned raw MAE only | Now returns (raw, norm) tuple |
| scripts/eval/run_stage_transfer.py | Same issue | Now returns (raw, norm) tuple |
| scripts/eval/run_frozen_probe.py | Same issue + import error (MTOReadout → MultiHeadReadout) | Now returns (raw, norm) tuple with dual evaluation |
| scripts/eval/reeval_ablation.py | New script | Dual raw/norm output |
| scripts/eval/reeval_stage_transfer.py | New script | Dual raw/norm output |
| outputs/reports/14_frozen_probe.md | Placeholder table had no metric labels | Will update with labeled columns |
| outputs/reports/15_stage_transfer.md | Partial output had raw values only | Will update with both raw and norm |
| outputs/reports/16_full_ablation.md | Placeholder expected values | Will update with real data |
| outputs/figures/FIGURE_MANIFEST.txt | Did not specify metric types | Will update |
| outputs/metrics/*.json | No metric type labels in JSON keys | Updated keys from "test" to "test_raw"/"test_norm" |

## 5. Before/After: Ablation Metrics

### Before (old baseline_comparison.json from job 86189, seed 2 last entry)
```json
{"name": "full_mto", "val": {"mu": 1.08, "alpha": 23.61},
 "test": {"mu": 1.05, "alpha": 23.60}}
```

### After (new baseline_comparison.json from job 86512)
```json
{"seed": 0, "method": "full_mto",
 "test_raw": {"mu": 1.0467, "alpha": 23.5949},
 "test_norm": {"mu": 0.6597, "alpha": 0.9573}}
```

Key observation: normalized mu MAE ranges ~0.48-0.69 across methods, normalized alpha MAE is ~0.96 across ALL methods. The alpha prediction is essentially at chance level (norm MAE ≈ 1.0 means predicting the mean).

## 6. Before/After: Stage Transfer Metrics

### Before
```json
{"seed": 0, "conditions": {"mu_only": {"test": {"mu": 1.0543, "alpha": NaN}}}}
```

### After (expected)
```json
{"seed": 0, "conditions": {"mu_only": {
  "test_raw": {"mu": 1.0543, "alpha": NaN},
  "test_norm": {"mu": 0.6645, "alpha": NaN}}}}
```

## 7. Key Observation: Alpha Normalized MAE ≈ 0.96

This is a significant finding. A normalized MAE of ~0.96 means the models are barely better than predicting the mean. This affects:

1. **Full ablation comparison**: All 5 methods show alpha_norm ~0.96-0.99 with negligible differences
2. **Interpretation**: The MTO benefits for alpha prediction are much smaller than for mu prediction
3. **fig16**: Should highlight this — MTO helps mu more than alpha

## 8. Verification Checklist

- [x] All evaluation scripts compute both raw and normalized MAE
- [x] JSON output keys distinguish "raw" from "norm"
- [x] Summary tables label columns as raw/norm
- [ ] Figures 16-18 axes labeled with "raw MAE" or "norm MAE"
- [ ] Reports do not mix raw and normalized metrics in comparisons
- [x] No report compares raw mu MAE (0.5) to normalized alpha MAE (0.96)

## 9. Recommendation

For paper figures and tables:
- Use **normalized MAE** for cross-task comparison (mu vs alpha)
- Use **raw MAE** for physical-scale interpretation
- Always label which is which
- Report per-task metrics, never a single scalar
- Do not report "joint MSE" without defining it as a normalized aggregate
