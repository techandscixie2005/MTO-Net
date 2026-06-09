# Stage C Smoke Test Report

**Report**: 18_stage_c_smoke_report.md  
**Date**: 2026-06-09  
**Job**: 85918 (A800 GPU, ~2 min)

## Configuration
- Stage: C (mu, alpha, IR, Raman, UV)
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
| 0 | 3.6507 | 3.4205 |
| Best | - | **2.9686** |

- Loss decreased: YES
- All tasks trained: mu, alpha, ir (50 bins), raman (50 bins), uv (50 bins)
- No NaN or numerical issues

### Outputs
- Checkpoint: `outputs/smoke_stage_c/checkpoints/stage_c_seed101/best.pt`
- MTO cache: `outputs/smoke/cache/mto_cache_seed101.npz`
- MTO figure: `outputs/figures/debug/smoke/smoke_mto_map.png`

### Spectral Loaded
- IR: 16/16 molecules indexed
- Raman: 16/16 molecules indexed
- UV: 16/16 molecules indexed
- Normalization: UV mean=0.015, std=0.050 (small values, may benefit from log transform or minmax)

## Status: PASSED
Stage C forward, backward, training, cache, and visualization all work correctly.
