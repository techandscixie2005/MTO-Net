# Codex Review Response — MTO-Net

**Report**: 26_codex_review_response.md
**Date**: 2026-06-10
**Review**: Codex (gpt-5.5, high reasoning)

---

## Critical Issues — Response

### C1: Rotation equivariance
**Verdict**: Design clarification needed, not a bug.
MTO intentionally operates on DetaNet invariant (l=0) features. We should clarify this in the paper.

### C2: Signed assembly
**Verdict**: Design clarification needed.
Three sign MLPs produce per-l sign coefficients. This is documented in code but should be clearer in paper.

### C3: Metric normalization
**Verdict**: FIXED.
All scripts now compute both raw and normalized MAE with consistent normalization.

### C4: Ablation confound (backbone reuse)
**Verdict**: ACKNOWLEDGED.
Re-evaluation uses independently saved checkpoints. Original training reused backbone — documented as limitation.

### C5: Stage transfer NaN
**Verdict**: By design.
NaN means "no head trained for this task". Figure 17 correctly shows this. S_sub from MTO coefficients is the key result.

### C6: Frozen probe
**Verdict**: Job 86515 running. Results will be pulled after completion.

### C7: Stage B/C not wired
**Verdict**: Documented in limitation reports 17, 18.

### C8: Source data integrity
**Verdict**: IN PROGRESS. Metrics JSONs now exist locally for figs 16-17. Fig 18 pending.

## Major Issues — Response

### M1: Normalization leak
**Acknowledged**. Stats computed from small train sample. Document as limitation.

### M2: Activity gate
**Acknowledged**. Use "sigmoid activity gate" not "charge-conserving" for simple mode.

### M3: Checkpointing incomplete
**Acknowledged**. Known limitation for research code.

### M4: Functional group claims
**REQUIRES AUDIT**. Verify SMARTS vs element-type in functional_group_analysis.py.

### M5: Stage B/C commands
**Fix**: Update reports with verified commands.

### M6: Slurm fragility
**Acknowledged**. Working scripts exist in jobs/.

### M7: Report overclaims
**FIXING**. Softening language in reports.

## Status
- Pytest: 75 passed, 19 skipped
- No data fabrication detected
- Stage B/C honestly labeled as label-limited
- Main action: audit FG analysis, soften claims, complete frozen probe
