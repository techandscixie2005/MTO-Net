# Implementation Summary — MTO-Net

**Date:** 2026-06-09
**Branch:** tensor-mto-v2
**Commit:** aaccf1a (Stage B/C code completion)

## What Was Built

### Core MTO-Net Code (`src/mto/`) — 27 Python files

**Adapter & Model:**
- `detanet_adapter.py` — Wraps DetaNet, exposes atom-level scalar features
- `mto_model.py` — Full MTONet: backbone + MTO + readout
- `mto_module.py` — Signed center-free valence-adaptive MTO assembly
- `mto_readout.py` — Multi-head readout (mu, alpha, IR, Raman, UV)

**Modules:**
- `valence.py` — Valence electron counter (H, C, N, O, F)
- `activity_gate.py` — Activity gate (none, simple, fermi_dirac modes)
- `modules/response_heads.py` — Spectral readout heads

**Losses & Training:**
- `losses.py` — Composite loss (MSE + cosine spectral + diversity)
- `training.py` — Trainer with configurable LR, theta schedule

**Analysis & Visualization:**
- `analysis/mto_cache.py`, `analysis/functional_groups.py`, `analysis/frozen_probe.py`
- `stability.py`, `intervention.py`, `visualization.py`

**Tests:** 22 test files, 75 passing, 19 skipped (HPC-only spectral data)

### What Works
1. Full code path: DetaNet -> MTO -> Readout -> Loss -> Backprop
2. Stage A training: 5 seeds, 50 epochs, best seed test MSE 0.448
3. Stage B/C smoke tests: code path verified on HPC
4. Seed subspace stability: S_sub ~0.51 +/- 0.07
5. Chemical validation: functional group enrichment detected
6. Slot intervention: property specialization confirmed
7. 15 paper figures (PNG + PDF) for all Stage A results

### What Remains (Blocked on HPC/SSH)
1. Full Stage B/C training on QM9S
2. Baseline/ablation full runs
3. Frozen probe experiment
4. Figures 16-18
5. Final git commit + push + server sync
