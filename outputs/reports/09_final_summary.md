# MTO-Net Final Project Summary

**Date:** 2026-06-09
**Branch:** `tensor-mto-v2`
**Commit:** pending (after final changes)
**Status:** Stage A complete, Stage B/C code complete, baseline infrastructure ready

## 1. Project Identity

- **Project:** MTO-Net: Molecular Tensor Orbital Network
- **Title:** Valence-Adaptive Molecular Tensor Orbitals for Equivariant Molecular Response Learning
- **GitHub:** `git@github.com:techandscixie2005/MTO-Net.git`
- **Local path:** `/home/xiangyu_xie/MTO`
- **Server path:** `/data/home/scwc008/run/xxy/MTO` (bjhpc_xxy_1)

## 2. Environment

- **Local:** Python 3.10 (conda: graduation_project), PyTorch 2.8.0+cu126, RTX 3050 Ti
- **Server:** Python 3.10 (conda: py310-torch270-vllm090), PyTorch 2.7.0+cu128, A800 GPUs
- **HPC hostname:** ln02
- **Workspace:** `/data/home/scwc008/run/xxy`
- **Slurm partition:** gpu_a800

## 3. Dataset

- **QM9S:** 129,817 molecules in `qm9s.pt`
- **Available labels:** dipole (mu), polarizability (alpha)
- **Missing in .pt:** IR spectrum, Raman spectrum, UV spectrum, Hessian, frequencies, normal modes, NMR
- **Spectral CSVs (HPC only):** ir_boraden.csv (3501 bins), raman_boraden.csv (3501 bins), uv_boraden.csv (701 bins)

## 4. Enabled Tasks

| Task | Stage | Labels in .pt? | CSV Available? | Enabled? |
|------|-------|---------------|----------------|----------|
| mu (dipole) | A | Yes | - | **Enabled** |
| alpha (polarizability) | A | Yes | - | **Enabled** |
| ir (IR spectrum) | B | No | Yes (CSV) | **Enabled** (direct spectrum regression) |
| raman | B | No | Yes (CSV) | **Enabled** (direct spectrum regression) |
| uv (UV spectrum) | C | No | Yes (CSV) | **Enabled** (direct spectrum regression) |
| hessian | B | No | No | **Skipped** — no labels |
| frequencies | B | No | No | **Skipped** — no labels |
| normal_modes | B | No | No | **Skipped** — no labels |
| nmr | C | No | No | **Skipped** — no labels |

**Note:** Stage B/C use direct spectrum regression from CSV files rather than the preferred physical-quantity-to-spectrum pipeline, because Hessian/normal modes/dipole derivatives are not available in QM9S .pt file.

## 5. Tests

- **Status:** 75 passed, 19 skipped, 0 failures
- **Skipped:** Spectral CSV tests (data on HPC only)
- **Coverage:** Core modules tested (valence, shapes, mask, forward, checkpoint, permutation, translation, rotation, data splits)

## 6. Smoke Test Results

| Stage | Job ID | GPU | Molecules | Epochs | Status |
|-------|--------|-----|-----------|--------|--------|
| A | 85594 (1 seed) | A800 | 5000 | 50 | Complete |
| A smoke | local | CPU | 32 | 2 | Pass |
| B smoke | 85917 | A800 | 16 | 2 | Pass |
| C smoke | 85918 | A800 | 16 | 2 | Pass |

## 7. Full Training (Stage A)

**Job 85594** on bjhpc_xxy_1, A800 GPUs, 5 seeds, 50 epochs, batch_size=64

### Results

| Seed | Test MSE | mu MAE | alpha MAE | Status |
|------|----------|--------|-----------|--------|
| 0 | 0.860 | 0.672 | 0.197 | Complete (plateaued) |
| 1 | **0.448** | **0.424** | **0.150** | Complete (best) |
| 2 | 0.859 | 0.671 | 0.192 | Complete (plateaued) |
| 3 | **0.467** | **0.425** | **0.150** | Complete (good) |
| 4 | 1.082 | 0.746 | 0.221 | Complete (plateaued) |

**Mean seed subspace stability:** 0.51 ± 0.07

## 8. Stage B/C Full Training

**Not yet run.** Smoke tests pass. Full training requires HPC Slurm submission.
Configs, scripts, and code are ready at:
- `configs/train/stage_b_ir_raman.yaml`
- `configs/train/stage_c_uv.yaml`
- `scripts/train_stage.py` (supports --stage b, --stage c)

## 9. Checkpoint Paths (HPC server)

```
outputs/checkpoints/stage_a_seed0/best.pt
outputs/checkpoints/stage_a_seed1/best.pt  # Best seed
outputs/checkpoints/stage_a_seed2/best.pt
outputs/checkpoints/stage_a_seed3/best.pt
outputs/checkpoints/stage_a_seed4/best.pt
```

## 10. Figure Manifest

All figures generated under `outputs/figures/stage_a/` and copied to `outputs/figures/final/`:

| Figure | Description | Status |
|--------|-------------|--------|
| fig1 | Seed prediction performance | Complete |
| fig2 | Seed subspace stability heatmap | Complete |
| fig3 | Training dynamics (loss curves) | Complete |
| fig4 | MTO cache overview (activity) | Complete |
| fig5 | Good/bad MTO comparison | Complete |
| fig6 | Property-specific MTO usage | Complete |
| fig7 | mu/alpha specialization | Complete |
| fig8 | Representative MTO atom maps | Complete |
| fig9 | Good seed atom map consistency | Complete |
| fig10 | Good/bad atom map comparison | Complete |
| fig11 | Slot intervention | Complete |
| fig12 | Functional group enrichment overview | Complete |
| fig13 | mu/alpha functional group association | Complete |
| fig14 | Representative functional group maps | Complete |
| fig15 | Good/bad functional group enrichment | Complete |
| fig16-18 | Baselines, stage transfer, frozen probe | Pending |

## 11. Reports

| Report | Path | Status |
|--------|------|--------|
| 00_repo_audit | outputs/reports/00_repo_audit.md | Complete |
| 00_dataset_audit | outputs/reports/00_dataset_audit.md | Complete |
| 00_detanet_forward | outputs/reports/00_detanet_forward.md | Complete |
| 00_env | outputs/reports/00_env.md | Complete |
| 00_slurm_gpu | outputs/reports/00_slurm_gpu.md | Complete |
| 01_design | outputs/reports/01_design.md | Complete |
| 04_evaluation_metrics | outputs/reports/04_evaluation_metrics.md | Complete |
| 05_mto_stability | outputs/reports/05_mto_stability.md | Complete |
| 06_intervention | outputs/reports/06_intervention.md | Complete |
| 07_chemical_validation | outputs/reports/07_chemical_validation.md | Complete |
| 08_baselines_ablations | outputs/reports/08_baselines_ablations.md | Complete |
| 09_final_summary | outputs/reports/09_final_summary.md | Complete (this file) |

## 12. Known Limitations

1. **Only mu + alpha labels in QM9S .pt file.** No Hessian, frequencies, or normal modes available for physical-quantity-to-spectrum pipeline.
2. **Stage B/C full training not run** — only smoke tests completed (code is ready).
3. **Full baseline and ablation comparisons not run** — code infrastructure ready, needs HPC GPU time.
4. **MTO routing uses only scalar features** (not full equivariant tensor). This limits tensor-order-specific MTO assembly.
5. **DetaNet representation is flat** (scalar features + single irrep tensor), not explicit l=0,1,2 dictionaries.
6. **Seed variance** (3/5 seeds plateau early) suggests training instability at batch_size=64.
7. **SSH to bjhpc_xxy_1 unreachable** (2026-06-09) — blocks all HPC operations.
8. **Functional group analysis** limited by SMARTS coverage and data quality.
9. **No RDKit SMILES verification** for all molecules — connectivity assumptions may be wrong.

## 13. Next Steps

1. Restore SSH connectivity to bjhpc_xxy_1
2. Push current state to GitHub
3. Sync to server
4. Run baseline/ablation comparisons on HPC
5. Run full Stage B/C training with spectral CSV data
6. Generate remaining figures (fig16-18)
7. Compute full stability metrics across stages
8. Complete frozen probe experiment
9. Polish paper narrative

## 14. Safety Confirmation

All HPC operations remained inside `/data/home/scwc008/run/xxy`. No files outside workspace were modified. No destructive commands were run. Dataset directories were treated as read-only.
