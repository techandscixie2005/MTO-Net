# QM9S Spectral CSV Schema Audit Report

## IR Spectrum: `ir_boraden.csv`
- **File size**: 8329.1 MB
- **Data rows**: 129,817
- **Total columns**: 3502
- **First 3 rows** (first 5 intensity values each):
  - Row 0: mol_id=0, first intensities=['0', '0.005416360218077898', '0.005428798496723175', '0.005441281944513321', '0.005453811958432198']
  - Row 1: mol_id=1, first intensities=['1', '0.05511827394366264', '0.055347368121147156', '0.05557793378829956', '0.05580994114279747']
  - Row 2: mol_id=2, first intensities=['2', '0.003754683770239353', '0.0037615324836224318', '0.0037684014532715082', '0.0037752892822027206']
- **Molecule ID column**: ``
- **SMILES column**: None detected
- **Spectral grid**: 3501 bins, wide format
  - Grid range: 500.0 to 4000.0, step=1.0

### Alignment to qm9s.pt
- **pt_exists**: True
- **pt_size_gb**: 2.68
- **qm9s_pt_molecule_count**: 129818
- **csv_row_count**: 129817
- **aligned**: False
- **strategy**: NEEDS_INVESTIGATION
- **note**: Mismatch: CSV has 129817 rows, PT has 129818 molecules

### NaN / Data Quality
- Sampled rows: 2031
- Malformed rows: 0
- Columns containing NaN: 0/3502

### Normalization
- **mean**: 2.37010124769737
- **std**: 10.291220155660204
- **min**: 0.003754683770239353
- **max**: 178.39060974121094
- **range**: 178.3868550574407
- **needs_normalization**: yes (z-score recommended)
- **recommendation**: Apply per-task z-score normalization before training

---

## RAMAN Spectrum: `raman_boraden.csv`
- **File size**: 8377.6 MB
- **Data rows**: 129,817
- **Total columns**: 3502
- **First 3 rows** (first 5 intensity values each):
  - Row 0: mol_id=0, first intensities=['0', '0.002817572560161352', '0.002821581671014428', '0.002825601724907756', '0.0028296317905187607']
  - Row 1: mol_id=1, first intensities=['1', '0.0013019494945183396', '0.0013043386861681938', '0.0013067388208582997', '0.0013091498985886574']
  - Row 2: mol_id=2, first intensities=['2', '0.00045606133062392473', '0.00045656098518520594', '0.0004570617456920445', '0.00045756352483294904']
- **Molecule ID column**: ``
- **SMILES column**: None detected
- **Spectral grid**: 3501 bins, wide format
  - Grid range: 500.0 to 4000.0, step=1.0

### Alignment to qm9s.pt
- **pt_exists**: True
- **pt_size_gb**: 2.68
- **qm9s_pt_molecule_count**: 129818
- **csv_row_count**: 129817
- **aligned**: False
- **strategy**: NEEDS_INVESTIGATION
- **note**: Mismatch: CSV has 129817 rows, PT has 129818 molecules

### NaN / Data Quality
- Sampled rows: 2031
- Malformed rows: 0
- Columns containing NaN: 0/3502

### Normalization
- **mean**: 0.27970697570027553
- **std**: 1.0021884103696745
- **min**: 0.00045606133062392473
- **max**: 16.48785400390625
- **range**: 16.487397942575626
- **needs_normalization**: yes (z-score recommended)
- **recommendation**: Apply per-task z-score normalization before training

---

## UV Spectrum: `uv_boraden.csv`
- **File size**: 1467.7 MB
- **Data rows**: 129,817
- **Total columns**: 702
- **First 3 rows** (first 5 intensity values each):
  - Row 0: mol_id=0, first intensities=['0', '0.0', '0.0', '0.0', '0.0']
  - Row 1: mol_id=1, first intensities=['1', '0.0', '0.0', '0.0', '0.0']
  - Row 2: mol_id=2, first intensities=['2', '0.0', '0.0', '0.0', '0.0']
- **Molecule ID column**: ``
- **SMILES column**: None detected
- **Spectral grid**: 701 bins, wide format
  - Grid range: 1.0 to 15.0, step=0.020000000000000018

### Alignment to qm9s.pt
- **pt_exists**: True
- **pt_size_gb**: 2.68
- **qm9s_pt_molecule_count**: 129818
- **csv_row_count**: 129817
- **aligned**: False
- **strategy**: NEEDS_INVESTIGATION
- **note**: Mismatch: CSV has 129817 rows, PT has 129818 molecules

### NaN / Data Quality
- Sampled rows: 2031
- Malformed rows: 0
- Columns containing NaN: 0/702

### Normalization
- **mean**: 5.098016592391672e-07
- **std**: 5.9575374500323645e-06
- **min**: 0.0
- **max**: 0.00010763685713754967
- **range**: 0.00010763685713754967
- **needs_normalization**: likely_ok
- **recommendation**: Apply per-task z-score normalization before training

---

## Summary
| Spectrum | Rows | Bins | Format | Aligned | Size (MB) |
|----------|------|------|--------|---------|----------|
| IR | 129,817 | 3501 | wide | NO | 8329 MB |
| RAMAN | 129,817 | 3501 | wide | NO | 8378 MB |
| UV | 129,817 | 701 | wide | NO | 1468 MB |
