# Stage B/C Code Completion Report

**Report**: 19_stage_bc_code_completion_report.md  
**Date**: 2026-06-09  
**Git commit**: aaccf1a  
**Branch**: tensor-mto-v2

## 1. Dataset Completion Status

All QM9S Figshare dataset files present (25 GB):
- `qm9s.pt` (2.7 GB, 129,818 molecules)
- `ir_boraden.csv` (8.2 GB, 129,817 rows, 3501 bins)
- `raman_boraden.csv` (8.2 GB, 129,817 rows, 3501 bins)
- `uv_boraden.csv` (1.5 GB, 129,817 rows, 701 bins)

Status: **COMPLETE** — all files verified.

## 2. Spectral CSV Schema Summary

| Spectrum | Bins | Range | Format | Rows | Aligned |
|----------|------|-------|--------|------|---------|
| IR | 3501 | 500-4000 cm⁻¹ | wide | 129,817 | mol_id lookup |
| Raman | 3501 | 500-4000 cm⁻¹ | wide | 129,817 | mol_id lookup |
| UV | 701 | 1.0-15.0 eV | wide | 129,817 | mol_id lookup |

1-molecule gap: qm9s.pt has 129,818 molecules, CSVs have 129,817. Handled by mol_id-based lookup.

## 3. Alignment Strategy

mol_id-based alignment via the first CSV column (integer molecule index). Verified by matching counts and consistent mol_id ranges. Efficient subset loading implemented for smoke tests.

## 4. Code Files Added/Modified

### New files (21)
- `src/mto/data/__init__.py`, `src/mto/data/qm9s_spectral.py` — spectral CSV loading
- `src/mto/modules/__init__.py`, `src/mto/modules/response_heads.py` — spectral readout heads
- `src/mto/plotting/plot_spectra.py` — spectrum visualization
- `scripts/audit/05_qm9s_spectral_csv_audit.py` — CSV schema audit
- `scripts/slurm/smoke_stage_b.sh`, `smoke_stage_c.sh` — smoke test Slurm scripts
- `configs/eval/stage_b_eval.yaml`, `stage_c_eval.yaml` — evaluation configs
- `configs/train/stage_b_smoke.yaml`, `stage_c_smoke.yaml` — smoke configs
- `outputs/reports/14-18_*.md` — 5 reports
- `tests/test_qm9s_spectral_schema.py`, `test_spectral_alignment.py`, `test_spectral_loss.py`, `test_spectral_cache_schema.py`, `test_spectrum_plotting.py`, `test_stage_b_forward.py`, `test_stage_c_forward.py` — 7 test files

### Modified files (6)
- `.gitignore` — added smoke/slurm exclusions
- `src/mto/dataset_qm9s.py` — spectral collator, subset loading
- `src/mto/mto_model.py` — spectral task support
- `src/mto/valence.py` — extended valence table to Z≤9
- `scripts/train_stage.py` — spectral loading, Stage B/C
- `configs/train/stage_b_ir_raman.yaml`, `stage_c_uv.yaml` — updated

## 5. Tests Run

**Unit/forward tests**: 21/21 passed  
**Schema/alignment tests**: 18/18 passed  
**Total**: 39 passed, 0 failed

## 6. Stage B Smoke Result
- Job 85917, A800 GPU, 16 mols, 2 epochs
- Val loss: 2.60 (decreasing from 4.63)
- Status: **PASSED**

## 7. Stage C Smoke Result
- Job 85918, A800 GPU, 16 mols, 2 epochs
- Val loss: 2.97 (decreasing from 3.42)
- Status: **PASSED**

## 8. Debug Figure Paths
- `outputs/figures/debug/smoke/smoke_mto_map.png`
- `outputs/figures/debug/smoke/smoke_mto_map.pdf`

## 9. Stage A Job Status
- Job 85594 (4 seeds) completed; interpretability job 85753 completed
- No running Stage A jobs

## 10. Git Branch
`tensor-mto-v2`

## 11. Local Commit Hash
`aaccf1a`

## 12. GitHub Push Status
- Pushed to `git@github.com:techandscixie2005/MTO-Net.git`
- Branch: `tensor-mto-v2`
- Status: **SUCCESS**

## 13. Large Files Excluded
All data files, checkpoints, caches, and large outputs excluded via .gitignore and rsync filters.

## 14. Remaining Limitations
1. **Spectral prediction quality**: 2-epoch smoke test verifies code correctness, not prediction quality. Full training needed.
2. **Normalization**: UV values are very small (std~5e-6), may need log transform or minmax scaling.
3. **Direct spectrum regression**: No physical derivative-based construction. Spectral CSVs used as response-level supervision.
4. **1 molecule gap**: mol_id 129,817 has no spectral data.
5. **Downsampling**: Smoke tests used 50-bin downsampled spectra; full resolution (3501/701 bins) needs more GPU memory.
6. **No seed stability analysis for Stage B/C yet**.

## 15. Next Recommended Step
Run full Stage B training on QM9S (100 epochs) initializing from Stage A best checkpoint, then proceed to seed stability and stage transfer analysis.
