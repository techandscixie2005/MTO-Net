# Baselines and Ablations — Plan and Partial Results

**Date:** 2026-06-09

## 1. Tier-1 Baselines (Required)

| Baseline | Description | Status |
|----------|-------------|--------|
| DetaNet direct readout | DetaNet backbone -> sum pool -> MLP readout | Code ready, runs pending |
| Positive attention pooling | Learned attention weights (no sign) -> pool -> MLP | Code ready, runs pending |
| MTO without signed coefficient | Softmax-only routing, c_ki = a_ki normalized | Code ready, runs pending |
| Fixed-K MTO | K = 20 fixed slots (no valence adaptivity) | Code ready, runs pending |
| Full MTO-Net | Signed, valence-adaptive, gated MTO | Stage A completed (5 seeds) |

## 2. Tier-2 Baselines (Optional)

| Baseline | Description | Status |
|----------|-------------|--------|
| No activity gate | g_k = 1.0 always | Code ready |
| No diversity regularization | λ_div = 0 | Code ready |
| Multi-token pooling | 4 independent K=5 token banks | Not implemented |
| Direct spectrum regression | MLP directly on atom pool without MTO | Not implemented |
| Frozen readout-only | Freeze backbone+MTO, train readout only | Code ready, runs pending |

## 3. Comparison Dimensions

Baselines are compared on:
1. **Prediction metrics:** Test MSE, MAE for mu + alpha
2. **Seed subspace stability:** S_sub across seeds
3. **Functional-group enrichment:** Carbonyl, aromatic enrichment scores
4. **Slot intervention selectivity:** Property specialization of slots
5. **Training efficiency:** Memory, wall-clock time, epochs to converge

## 4. Full MTO vs Baselines (Expected)

Based on Stage A results for Full MTO (seed 1: test MSE 0.448, mu MAE 0.424, alpha MAE 0.150):

### Predicted comparisons:

| Metric | Full MTO | Direct Readout | Attn Pool | No-sign MTO | Fixed-K |
|--------|----------|---------------|-----------|-------------|---------|
| mu MAE | 0.424 | similar/~0.42 | similar | +0.02-0.05 | +0.03-0.07 |
| alpha MAE | 0.150 | similar/~0.15 | similar | +0.01-0.03 | +0.02-0.05 |
| S_sub | 0.51 | N/A (no slots) | 0.15-0.30 | 0.35-0.45 | 0.30-0.40 |
| Carbonyl enrichment | 1.42 | N/A | 0.95-1.05 | 1.10-1.20 | 1.05-1.20 |
| Slot specialization | Yes | N/A | No | Partial | Partial |

### Expected key result:
> Full MTO matches or exceeds baseline prediction accuracy while providing stronger stability and interpretability signals.

## 5. Ablation: Sign Coefficient Effect

Removing the signed coefficient (c_ki = a_ki normalized, positive only) is expected to:
- Slightly reduce or maintain prediction accuracy
- **Significantly reduce** subspace stability (no bonding/antibonding organization)
- **Reduce** functional-group enrichment
- Make MTOs behave more like ordinary attention

This ablation is the strongest test of the signed assembly contribution.

## 6. Ablation: Fixed-K Effect

Using K=20 fixed slots:
- Slightly worse prediction for large molecules (insufficient slots)
- Excess capacity for small molecules (unused/wasted slots)
- No valence-adaptivity intuition
- MTOs still form but less chemically organized

## 7. Implementation Status

- Baseline code: `src/mto/baselines.py` (attention pooling, direct readout, no-sign, fixed-K)
- Training scripts: `scripts/train_baseline.py`, `scripts/train_ablation.py`
- Eval scripts: `scripts/eval_baseline.py`
- Configs: `configs/ablation/`, `configs/train/detanet_baseline.yaml`

## 8. Remaining Work

Full baseline comparisons require HPC GPU runs. Code infrastructure is ready. Priority order:
1. Direct readout baseline (simplest, establishes prediction floor)
2. No-sign MTO ablation (tests core MTO innovation)
3. Fixed-K ablation (tests valence adaptivity)
4. Attention pooling baseline (tests signed vs unsigned)

## 9. Paths

- Baseline configs: `configs/ablation/`
- Baseline code: `src/mto/baselines.py`
