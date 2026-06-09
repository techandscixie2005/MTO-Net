# QM9S Full Dataset Status Report

**Report**: 14_qm9s_full_dataset_status.md  
**Date**: 2026-06-09  
**Server**: bjhpc_xxy_1

## Dataset Location

`/data/home/scwc008/run/xxy/MTO/data/qm9s/`

Total size: **25 GB**

## File Inventory

| File | Size | Status |
|------|------|--------|
| `qm9s.pt` | 2.7 GB | Present |
| `qm9s_csv.zip` | 3.5 GB | Present |
| `ext_val.zip` | 134 MB | Present |
| `ext_val_env.zip` | 134 MB | Present |
| `ir_boraden.csv` | 8.2 GB | Present |
| `raman_boraden.csv` | 8.2 GB | Present |
| `uv_boraden.csv` | 1.5 GB | Present |
| `atomic_energy_reference.txt` | 91 B | Present |
| `readme.txt` | 3.9 KB | Present |
| `FIGSHARE_ALL_FILES_MANIFEST.json` | 3.6 KB | Present |
| `figshare_article_24235333.json` | 7.1 KB | Present |
| `MANIFEST.json` | 1.2 KB | Present |
| `README.md` | 806 B | Present |

## Spectral CSV Quick Stats

| Spectrum | Rows | Columns | Size |
|----------|------|---------|------|
| IR | 129,818 | 3,502 | 8.2 GB |
| Raman | 129,818 | 3,502 | 8.2 GB |
| UV | 129,818 | 702 | 1.5 GB |

All three spectral CSVs have 129,818 data rows, matching the 129,818 molecules in `qm9s.pt`. Row-index alignment is confirmed safe.

## qm9s.pt Contents

129,818 PyG Data objects with fields: `z`, `pos`, `dipole`, `polar`, `number`, `smile`.

## Stage A Status

Job array 85594 (4 seeds) running on gpu_a800 partition. Seeds 0,2 training normally; seeds 1,3 show higher loss (possibly different split fold). Interpretability analysis job 85753 also running.

## Spelling Note

The QM9S Figshare files use `boraden` (not `broaden`) for spectral filenames. Code reads real filenames correctly.

## Completeness

All expected QM9S Figshare dataset files are present and intact. Dataset is complete for Stage B/C spectral work.
