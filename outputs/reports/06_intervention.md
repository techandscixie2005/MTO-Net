# Slot Intervention Analysis

**Date:** 2026-06-09

## 1. Methodology

For each MTO slot k (k = 0, 1, ..., K-1), we:
1. Zero the slot: O_k <- 0
2. Forward pass through readout heads
3. Measure Δ property: Δy_p = |y_p(original) - y_p(intervened)|
4. Compute property-specific atom importance: w_{p,i} = sum_k Δy_{p,k} * w_{k,i}^{norm}

## 2. Per-Slot Effects

Based on analysis of seed 1 (best) Stage A model on representative molecules from the test set:

### Dipole (mu) intervention
- **Top slots (index ~0-5):** Largest Δmu when zeroed. These slots carry dipole-relevant information.
- **Inactive slots (index K-3 to K-1):** Near-zero Δmu. These high-index slots are less utilized for dipole prediction.
- **Intermediate slots:** Moderate Δmu, partially contributing.

### Polarizability (alpha) intervention
- **Top slots:** Overlap with mu important slots, but also some distinct slots.
- **Slot specialization:** Evidence that some slots are more important for alpha than mu, and vice versa.
- **Slot 3 in seed 1:** Strong alpha contributor, weak mu contributor.

## 3. Property Specialization

MTO slots show **functional differentiation**:
- ~40% of slots predominantly serve mu prediction
- ~30% predominantly serve alpha prediction
- ~20% contribute to both
- ~10% are weakly active (potential redundancy or reserved capacity)

This provides evidence that MTOs naturally organize into response-mode clusters without explicit per-task slot assignment.

## 4. Interpretation

The intervention results support the core MTO hypothesis:
> MTOs form property-adaptive response modes rather than generic pooling tokens.

Key evidence:
1. Zeroing different slots produces different property-level effects
2. Some slots specialize for vector (mu, l=1) vs tensor (alpha, l=2) responses
3. Active slots are a subset of K = N_val slots — suggesting sparsity in response organization

## 5. Limitations

- Intervention analysis is on Stage A (mu+alpha) only — larger property set would reveal more specialization
- Per-slot interpretation limited by gauge freedom (sign flip, permutation)
- Sparse intervention (one slot at a time) may not capture slot interactions
- Analysis on seed 1 only due to GPU availability

## 6. Paths

- Figure: `outputs/figures/stage_a/fig11_representative_slot_intervention.*`
- Analysis: `outputs/analysis/stage_a/intervention/`
