# HPC Job Recovery Report

**Report**: 30_hpc_job_recovery.md
**Date**: 2026-06-10
**HPC Server**: bjhpc_xxy_1 (ln02)
**Workspace**: /data/home/scwc008/run/xxy/MTO

---

## 1. Previous Session Jobs (Verified Status)

| Job ID | Name | sacct State | ExitCode | Wall Time | Actual Outcome |
|--------|------|-------------|----------|-----------|----------------|
| 86187 | mto_transfer | COMPLETED | 0:0 | ~1.5h | COMPLETED — 3 seeds × 3 conds trained, but raw metrics only, subspace sim NULL |
| 86189 | mto_ablation | COMPLETED | 0:0 | ~2.5h | COMPLETED — 3 seeds × 5 methods trained 20 epochs, but only printed epoch 0, metrics RAW |
| 86186 | mto_frozen | FAILED | 1:0 | 24s | FAILED — RuntimeError: size mismatch (hidden_dim 64 checkpoint vs 32 model) |
| 86198 | mto_frozen | COMPLETED | 0:0 | 22s | FAILED — same RuntimeError, exit code 0 misleading (error on stderr) |

## 2. Root Cause Analysis

### Frozen Probe (86186/86198)
- **Root cause**: `run_frozen_probe.py` instantiates MTONet with `mto_hidden_dim=32, readout_hidden_dim=64` but Stage A checkpoint (seed 1) has `mto_hidden_dim=64, readout_hidden_dim=128`
- **Fix**: Updated `run_frozen_probe.py` to use `mto_hidden_dim=64, readout_hidden_dim=128`
- **Re-submitted**: Job 86513

### Ablation (86189)  
- **Training**: Actually trained 20 epochs per the Slurm script `--epochs 20` — but the code only printed epoch 0
- **Metrics**: `evaluate_model()` computed RAW MAE only (alpha ~23.6, mu ~1.05)
- **Data loss**: Results JSON only contains last seed's data (overwrite each seed iteration)
- **Fix**: Updated `run_baselines.py` to print every 5 epochs, compute both raw and normalized MAE. Created `reeval_ablation.py` for checkpoint-based re-evaluation.
- **Re-submitted**: Job 86512 (re-eval only, no re-training needed)

### Stage Transfer (86187)
- **Training**: Trained 20 epochs per Slurm script
- **Subspace similarity**: NULL — `compute_mto_subspace()` looked for `model.mto.coeffs` which doesn't exist; proper access is `model(..., return_mto=True)["coeff"]`
- **Metrics**: RAW MAE only
- **Fix**: Fixed `compute_mto_subspace()` to use `return_mto=True`, updated `evaluate_model()` for dual raw/norm metrics. Created `reeval_stage_transfer.py`.
- **Re-submitted**: Job 86514 (re-eval only)

## 3. New Jobs Submitted

| Job ID | Name | Type | Expected Duration | Dependencies |
|--------|------|------|-------------------|-------------|
| 86512 | reeval_abl | Re-evaluation (GPU) | ~30 min | None — uses existing checkpoints |
| 86513 | mto_frozen2 | Full training (GPU) | ~2-3 hours | None — trains from scratch |
| 86514 | reeval_st | Re-evaluation (GPU) | ~30 min | None — uses existing checkpoints |

## 4. Monitoring Commands

```bash
# Check all user jobs
ssh bjhpc_xxy_1 'squeue -u scwc008'

# Check specific jobs
ssh bjhpc_xxy_1 'squeue -j 86512,86513,86514'

# Check job history
ssh bjhpc_xxy_1 'sacct -j 86512,86513,86514 --format=JobID,JobName,State,ExitCode,Elapsed -P'

# Monitor logs
ssh bjhpc_xxy_1 'tail -f outputs/logs/slurm/reeval_abl_86512.out'
ssh bjhpc_xxy_1 'tail -f outputs/logs/slurm/frozen_probe2_86513.out'
ssh bjhpc_xxy_1 'tail -f outputs/logs/slurm/reeval_st_86514.out'
```

## 5. Files Modified/Synced to HPC

| Local File | HPC Path | Changes |
|-----------|----------|---------|
| scripts/eval/run_frozen_probe.py | synced | mto_hidden_dim 32→64, readout_hidden_dim 64→128 |
| scripts/eval/run_baselines.py | synced | Print every 5 epochs, dual raw/norm metrics in evaluate_model, fixed summary |
| scripts/eval/run_stage_transfer.py | synced | Fixed compute_mto_subspace, dual raw/norm metrics, fixed summary |
| scripts/eval/reeval_ablation.py | synced (new) | Checkpoint re-evaluation with normalized metrics |
| scripts/eval/reeval_stage_transfer.py | synced (new) | Checkpoint re-evaluation with normalized metrics + subspace sim |
| jobs/reeval_ablation.slurm | created on HPC | Slurm job for ablation re-eval |
| jobs/frozen_probe_fixed.slurm | created on HPC | Slurm job for fixed frozen probe |
| jobs/reeval_stage_transfer.slurm | created on HPC | Slurm job for stage transfer re-eval |

## 6. GPU Resources

```
GPU available: 1× NVIDIA A800-SXM4-80GB
Partition: gpu_a800
```

All three jobs share the single A800 GPU — they will run sequentially in Slurm queue.
