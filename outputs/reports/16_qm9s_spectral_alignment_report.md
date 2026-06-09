# QM9S Spectral Alignment Report

**Report**: 16_qm9s_spectral_alignment_report.md  
**Date**: 2026-06-09

## Alignment Strategy

### Molecule Identification

All three spectral CSV files (`ir_boraden.csv`, `raman_boraden.csv`, `uv_boraden.csv`) use **integer mol_id** as the first column (column header is empty string `""`). The mol_id matches the `number` attribute in `qm9s.pt` PyG Data objects.

### Row Count Discrepancy

- **qm9s.pt**: 129,818 molecules
- **Spectral CSVs**: 129,817 data rows each (mol_ids 0 through 129,816)

The last molecule (mol_id 129,817) in qm9s.pt lacks spectral data. This is confirmed by: all CSVs have identical row counts, all have mol_id range 0-129,816, and the count is off by exactly 1.

### Alignment Method

**mol_id-based lookup** is the primary alignment method:

1. Parse each spectral CSV into a dict `{mol_id: [intensities]}` using streaming CSV reading
2. For each molecule from qm9s.pt, look up its `mol_id` in the spectral index
3. If found, attach spectral target; if not found (mol 129,817 only), skip

This handles the 1-molecule gap cleanly.

### Alignment Verification

| Check | Result |
|-------|--------|
| All 3 CSVs have identical row count | PASS (129,817) |
| mol_id range consistent (0 to 129,816) | PASS |
| mol_id starts at 0 | PASS |
| Each row has correct bin count | PASS (IR: 3501, Raman: 3501, UV: 701) |
| First molecules have non-zero spectra | PASS |
| No NaN values in sampled rows | PASS |

### Data Format

- **Format**: Wide format (each row = one molecule, columns = spectral bins)
- **IR**: 3501 bins, 500-4000 cm⁻¹, step 1.0
- **Raman**: 3501 bins, 500-4000 cm⁻¹, step 1.0
- **UV**: 701 bins, 1.0-15.0 eV, step 0.02

### Normalization

All spectral tasks require per-task z-score normalization:
- IR: mean≈2.37, std≈10.29
- Raman: mean≈0.28, std≈1.00
- UV: mean≈5.1e-7, std≈5.96e-6 (very small values, minmax or log transform may be better)

### Limitations

1. **1 molecule gap**: mol_id 129,817 has no spectral data. Handled by skipping.
2. **No SMILES in CSV**: Spectral CSVs do not contain SMILES — alignment by mol_id only.
3. **Large file size**: IR/Raman CSVs are 8.2 GB each. Must use streaming reads, never load fully into memory.
4. **Row-order alignment**: Row order in CSV matches mol_id order, but code uses explicit mol_id lookup (not order-dependent).
