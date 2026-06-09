# Implementation Plan — MTO-Net

**Date:** 2026-06-09

## Phases

### Phase 0: Audit (COMPLETE)
- Inspect environment, repo, data, GPU/Slurm
- Identify DetaNet hook point
- Document dataset labels
- Output: 00_repo_audit.md, 00_env.md, 00_dataset_audit.md, 00_detanet_forward.md, 00_slurm_gpu.md

### Phase 1: Core Implementation (COMPLETE)
1. DetaNet adapter (`src/mto/detanet_adapter.py`)
2. Valence counter (`src/mto/valence.py`)
3. MTO module (`src/mto/mto_module.py`)
4. Activity gate (`src/mto/activity_gate.py`)
5. Readout heads (`src/mto/mto_readout.py`, `src/mto/modules/response_heads.py`)
6. Losses (`src/mto/losses.py`)
7. Full model (`src/mto/mto_model.py`)
8. Dataset loader (`src/mto/dataset_qm9s.py`, `src/mto/data/qm9s_spectral.py`)
9. Training loop (`src/mto/training.py`, `scripts/train_stage.py`)
10. Config system (`configs/`)

### Phase 2: Tests (COMPLETE, 75 pass)
- Core: valence, shapes, mask, forward smoke, checkpoint
- Invariance: permutation, translation, rotation
- Data: task registry, dataset split, config loading
- Stage B/C: forward, loss, cache schema
- Spectral: schema, alignment, plotting

### Phase 3: Smoke Tests (Stage A COMPLETE, B/C PASS on 16-mol smoke)
- Local CPU smoke
- HPC Slurm smoke

### Phase 4: Stage A Full Training (COMPLETE, 5 seeds)
- Job 85594 on A800 GPUs

### Phase 5: Analysis & Figures (MOSTLY COMPLETE)
- MTO atom maps
- Seed subspace stability
- Slot intervention
- Functional group enrichment
- 15 figures generated

### Phase 6: Stage B/C Full Training (PENDING — blocked on SSH/HPC)
- Configs and scripts ready

### Phase 7: Baselines (PENDING — blocked on SSH/HPC)
- Code ready, needs HPC GPU runs

### Phase 8: Paper Reports (IN PROGRESS)
- Reports being written

## File Map

| Component | File | Status |
|-----------|------|--------|
| Adapter | src/mto/detanet_adapter.py | Complete |
| Valence | src/mto/valence.py | Complete |
| MTO module | src/mto/mto_module.py | Complete |
| Activity gate | src/mto/activity_gate.py | Complete |
| Readouts | src/mto/mto_readout.py | Complete |
| Response heads | src/mto/modules/response_heads.py | Complete |
| Full model | src/mto/mto_model.py | Complete |
| Losses | src/mto/losses.py | Complete |
| Training | src/mto/training.py | Complete |
| Dataset | src/mto/dataset_qm9s.py | Complete |
| Spectral data | src/mto/data/qm9s_spectral.py | Complete |
| Baselines | src/mto/baselines.py | Complete |
| Configs | configs/train/, configs/eval/, configs/ablation/ | Complete |
| Training scripts | scripts/train_stage.py | Complete |
| Eval scripts | scripts/eval/*.py | Complete |
| Slurm jobs | jobs/*.slurm | Complete |
| Tests | tests/test_*.py (22 files) | 75 pass, 19 skip |
