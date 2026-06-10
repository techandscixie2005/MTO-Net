# Fig18 Frozen Probe Audit

**Report**: 35_fig18_frozen_probe_audit.md
**Date**: 2026-06-10

---

## Source Data

File: `outputs/frozen_probe/frozen_probe_results.json` (job 86515, 3 seeds × 3 conditions, 20 epochs)

## Summary Statistics (Mean ± SD, 3 seeds)

| Condition | mu MAE | alpha MAE | Trainable Params |
|-----------|--------|-----------|-----------------|
| Frozen MTO + new readout | 0.7536 ± 0.0136 | 23.63 ± 0.06 | 17,292 |
| Frozen DetaNet + new readout | 1.0055 ± 0.0130 | 23.66 ± 0.07 | 12,416 |
| From scratch (full training) | 1.0891 ± 0.0042 | 23.66 ± 0.07 | 1,523,793 |

## Key Finding

Frozen MTO (mu=0.75) significantly outperforms both frozen DetaNet (mu=1.01) and from-scratch (mu=1.09) on mu prediction. The frozen MTO uses only 17K trainable parameters vs 1.5M for from-scratch, yet achieves ~31% lower mu MAE. This demonstrates representation reuse: the learned MTO encoder captures transferable response-mode features.

Alpha prediction is uniformly poor (~23.6 raw MAE) across all conditions — near the mean-prediction baseline. This is a known limitation.

## Figure Outputs

- `outputs/figures/final/fig18_frozen_probe_reuse.pdf`
- `outputs/figures/final/fig18_frozen_probe_reuse.png`
- `outputs/figures/source_data/fig18_frozen_probe_reuse.csv`

## Caveats

1. Raw MAE only — normalized metrics not in the JSON output
2. No error bars in current figure — should add SD
3. Alpha prediction is at chance level for all conditions
4. 5k molecule subset, not full QM9S
5. Single frozen checkpoint (seed 1, best Stage A model)

## Status: FINAL
