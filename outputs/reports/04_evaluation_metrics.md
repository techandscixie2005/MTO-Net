# Evaluation Metrics — MTO-Net Stage A (mu + alpha)

**Date:** 2026-06-09
**Dataset:** QM9S full (129,817 molecules, split: 5000 train / 2000 val / 2000 test)

## 1. Stage A Full Run — 5 Seeds

Job 85594 on bjhpc_xxy_1, A800 GPUs, batch_size=64, 50 epochs.

### Val/Test MSE Summary

| Seed | Val MSE | Test MSE | mu MAE | alpha MAE | Best Epoch |
|------|---------|----------|--------|-----------|------------|
| 0 | 0.894 | 0.860 | 0.672 | 0.197 | ~10 |
| 1 | **0.440** | **0.448** | **0.424** | **0.150** | ~47 |
| 2 | 0.910 | 0.859 | 0.671 | 0.192 | ~10 |
| 3 | **0.448** | **0.467** | **0.425** | **0.150** | ~40 |
| 4 | 1.131 | 1.082 | 0.746 | 0.221 | ~10 |

### Interpretation

- **Good seeds (1, 3):** Continue improving throughout 50 epochs, reaching test MSE ~0.45
- **Bad seeds (0, 2, 4):** Plateau after epoch 10, stuck in poor local minimum
- **Seed variance:** Best seeds ~2x better than worst. Real optimization instability at batch_size=64
- **Mean test MSE across 5 seeds:** 0.743 (mu MAE 0.587, alpha MAE 0.182)

### Model parameters: 1,523,793

## 2. Stage B Smoke Test

### Smoke test (Job 85917, A800, 16 mols, 2 epochs)

| Metric | Value |
|--------|-------|
| Tasks | mu, alpha, ir, raman |
| Val loss | 2.598 (decreasing from 4.63) |
| IR bins | 3501 |
| Raman bins | 3501 |

Status: PASSED (code path verified)

## 3. Stage C Smoke Test

### Smoke test (Job 85918, A800, 16 mols, 2 epochs)

| Metric | Value |
|--------|-------|
| Tasks | mu, alpha, ir, raman, uv |
| Val loss | 2.972 (decreasing from 3.42) |
| UV bins | 701 |

Status: PASSED (code path verified)

## 4. Normalization

All labels z-score standardized (train split mean=0, std=1). Metrics reported on standardized scale. Restore for physical units in final reporting.

## 5. Data

- QM9S qm9s.pt: dipole + polarizability (no Hessian, frequencies, or spectral labels in .pt)
- Spectral CSV files (ir_boraden.csv, raman_boraden.csv, uv_boraden.csv) available on HPC server
- 1-molecule gap: qm9s.pt has 129,818 molecules, CSVs have 129,817

## 6. Paths

- Metrics: `outputs/metrics/stage_a_seed*.json`
- Val/test breakdown: `outputs/metrics/stage_a/stage_a_val_test_metrics.json`
- Checkpoint (seed 1 best): `outputs/checkpoints/stage_a_seed1/best.pt` (on HPC)
- Split: `outputs/splits/qm9s_split_stage_a.json`
