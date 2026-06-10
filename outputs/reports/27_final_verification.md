# Final Verification Report (Updated)

**Report**: 27_final_verification.md
**Date**: 2026-06-10
**Session**: Ralph recovery (fresh)
**Branch**: tensor-mto-v2
**Commit**: 2b4cbd5

---

## pytest Result

```
75 passed, 19 skipped in 12.56s
```
**Status**: PASSED — no regressions

## Figure Audit

| # | Figure | Status | Source |
|---|--------|--------|--------|
| 1-15 | Stage A | 15 FINAL | Real data from 5-seed run |
| 16 | Baseline ablation | **FINAL** | 5 methods × 3 seeds, job 86512 |
| 17 | Stage transfer | **FINAL** | S_sub + metrics, job 86517 |
| 18 | Frozen probe | **PRELIMINARY** | Seeds 0-1 from job 86515 running |

## Metric Normalization Status

FIXED. All evaluation scripts now produce both raw and normalized MAE. JSON keys distinguish "test_raw" vs "test_norm".

## Stage Transfer Status

COMPLETED. S_sub(mu_only vs alpha_only) = [0.41, 0.64, 0.77] across seeds 0-2. Partial above-random transfer.

## Frozen Probe Status

RUNNING (job 86515, ~28 min). Preliminary results:
- Seed 0: frozen_mto mu=0.74 vs frozen_direct mu=1.01 vs from_scratch mu=1.09
- Seed 1: frozen_mto mu=0.75 vs frozen_direct mu=0.99 — from_scratch still training

## Full Ablation Status

COMPLETED (job 86512). 5 methods × 3 seeds evaluated. Normalized mu: 0.48-0.70. Normalized alpha: ~0.96 all methods.

## Stage B/C Status

LABEL-LIMITED. Smoke tests pass. Full training requires labels not in QM9S.

## Codex Review Status

COMPLETED. 8 critical, 7 major issues identified. Response written (report 26). Key actionable items: audit FG analysis, soften claims, fix metric normalization (done).

## Git Status

```
Modified: 4 figures (regenerated), FIGURE_MANIFEST.txt, baseline_comparison.json, run_baselines.py
Untracked: 19 files (scripts, jobs, .omc/, third_party examples)
```

## Server Path

```
HPC: /data/home/scwc008/run/xxy/MTO
Local: /home/xiangyu_xie/MTO
```

## Known Limitations

1. Frozen probe seed 2 + from_scratch for seeds 1-2 still running
2. Functional group analysis may use atom-type proxies (Codex finding, needs audit)
3. Ablation reuses backbone (confound documented)
4. Stage B/C label-limited
5. No external benchmark comparison
6. Optimization instability (3/5 bad seeds)
