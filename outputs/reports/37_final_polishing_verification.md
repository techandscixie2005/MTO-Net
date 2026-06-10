# Final Polishing Verification Report

**Report**: 37_final_polishing_verification.md
**Date**: 2026-06-10
**Session**: Ralph polishing

---

## pytest Result

```
75 passed, 19 skipped in 13.08s
```
**Status**: PASSED — no regressions.

## Fig18 Status

**FINAL**. Publication-grade figure with nature-figure styling:
- Error bars (SD) across 3 seeds
- Individual seed markers
- Source CSV: `outputs/figures/source_data/fig18_frozen_probe_reuse.csv`

## Functional-Group Audit Status

**Case B confirmed**: Atom-type proxy only. RDKit not installed, SMARTS not used. Claims downgraded from "functional-group enrichment" to "atom-type enrichment" in claim-evidence table. Audit report: `34_functional_group_audit.md`.

## Claim-Evidence Table Status

**UPDATED**. Conservative wording applied:
- Claim 5: "atom-type enrichment (C, N, O, F)" not "functional group enrichment"
- Claim 7: Full 3-seed stats with mean±SD
- P1: Clarified as atom-type enrichment

## LaTeX Report

**Path**: `outputs/reports/36_mtonet_project_report.tex`
**PDF**: `outputs/reports/36_mtonet_project_report.pdf` (218 KB, compiled with pdflatex)

Contains all 20 required sections: Introduction, Scope, Theory, Code Architecture, Theory-Code Consistency, Dataset, Stage A Results, Seed Stability, Slot Intervention, Chemical Enrichment, Frozen Probe, Stage Transfer, Ablation, Stage B/C, Figures, Codex Review, Limitations, Next Steps, Data Availability, Conclusion.

## Git Status (Before Commit)

```
M outputs/figures/FIGURE_MANIFEST.txt
M outputs/figures/final/fig18_frozen_probe_reuse.pdf
M outputs/figures/final/fig18_frozen_probe_reuse.png
M outputs/reports/21_claim_evidence_table.md
?? outputs/reports/33_final_polishing_recovery.md
?? outputs/reports/34_functional_group_audit.md
?? outputs/reports/35_fig18_frozen_probe_audit.md
?? outputs/reports/36_mtonet_project_report.tex
?? outputs/reports/36_mtonet_project_report.pdf
?? outputs/reports/37_final_polishing_verification.md
?? outputs/figures/source_data/
```

## Remaining Limitations

1. Atom-type proxy (RDKit unavailable) — true functional-group validation pending
2. Alpha prediction at chance level (~0.96 normalized MAE)
3. 3/5 bad seeds in Stage A
4. No external benchmark comparison
5. Stage B/C label-limited (smoke tests only)
6. Frozen probe uses single Stage A checkpoint (seed 1)
7. Medium-scale (5k mols) for ablation/transfer/frozen probe
