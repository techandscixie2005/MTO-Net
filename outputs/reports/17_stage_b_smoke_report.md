# Stage B Smoke Test Report

**Report**: 17_stage_b_smoke_report.md  
**Date**: 2026-06-09  
**Job**: 85917 (A800 GPU, ~2 min)

## Configuration
- Stage: B (mu, alpha, IR, Raman)
- Molecules: 16 (subset)
- Epochs: 2
- Batch size: 4
- Feature dim: 64, MTO hidden: 32, Readout hidden: 64
- Activity gate: simple
- Spectral downsampling: 50 bins per spectrum
- GPU: NVIDIA A800-SXM4-80GB

## Results

### Loss
| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 0 | 3.9406 | 4.6308 |
| Best | - | **2.5950** |

- Loss decreased: YES
- All tasks trained: mu, alpha, ir (50 bins), raman (50 bins)
- No NaN or numerical issues

### Outputs
- Checkpoint: `outputs/smoke_stage_b/checkpoints/stage_b_seed100/best.pt`
- MTO cache: `outputs/smoke/cache/mto_cache_seed100.npz`
- MTO figure: `outputs/figures/debug/smoke/smoke_mto_map.png`

### Spectral Loaded
- IR: 16/16 molecules indexed
- Raman: 16/16 molecules indexed
- Normalization: z-score (IR mean=2.08, std=11.44; Raman mean=2.01, std=13.34)

## Status: PASSED
Stage B forward, backward, training, cache, and visualization all work correctly.
