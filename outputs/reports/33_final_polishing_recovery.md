# Final Polishing Recovery Report

**Report**: 33_final_polishing_recovery.md
**Date**: 2026-06-10
**Session**: Ralph polishing (fresh)

---

## Repository State

- **Branch**: tensor-mto-v2
- **Commit**: 05bcb3b — "fix: complete MTO-Net evidence experiments, regenerate figs 16-18 from real data"
- **Remote**: git@github.com:techandscixie2005/MTO-Net.git
- **Local path**: /home/xiangyu_xie/MTO
- **HPC path**: /data/home/scwc008/run/xxy/MTO

## Dirty Tree

```
M outputs/figures/final/fig_supp_stage_b_spectral_smoke.pdf
?? .omc/
?? 7 untracked scripts (analysis/interpretability/figure gen)
?? 4 untracked slurm scripts
?? 3 untracked DetaNet example files
```

## Final Figures (18 pairs PDF+PNG)

| # | Figure | Status |
|---|--------|--------|
| 1-15 | Stage A | FINAL |
| 16 | Baseline ablation | FINAL |
| 17 | Stage transfer | FINAL |
| 18 | Frozen probe | FINAL (needs nature-figure polish) |
| S | Stage B spectral smoke | PRELIMINARY (smoke only) |

## Key Reports (61 files)

All required reports from previous sessions exist. New reports to write this session: 33, 34, 35, 36, 37.

## Known Remaining Issues

1. Functional-group analysis is atom-type proxy (RDKit unavailable) — confirmed Case B
2. fig18 needs publication-grade polish with nature-skills:nature-figure
3. No LaTeX project report exists yet
4. Claim-evidence table still says "SMARTS-based" and "functional group enrichment" — needs correction
5. fig_supp_stage_b_spectral_smoke.pdf has uncommitted modification

## What This Session Will Fix

1. Audit functional-group analysis → confirmed atom-type proxy → update all claims
2. Polish fig18 with nature-figure → final publication-grade
3. Update claim-evidence table with corrected wording
4. Write comprehensive LaTeX project report
5. Run pytest verification
6. Commit and push
