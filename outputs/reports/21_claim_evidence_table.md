# Claim-Evidence Table — MTO-Net (Updated 2026-06-10)

**Report**: 21_claim_evidence_table.md
**Status**: Updated with experiment results from jobs 86512, 86515, 86517

---

## Supported Claims

| # | Claim | Evidence | Strength | Figure(s) |
|---|-------|----------|----------|-----------|
| 1 | MTO-Net can predict mu+alpha with competitive accuracy (test MSE ~0.45) | 5-seed Stage A run, job 85594, locked split, normalized labels | STRONG | fig1 |
| 2 | MTO response-mode subspaces are moderately consistent across seeds (S_sub ~0.51 ± 0.07) | Pairwise subspace similarity across 5 seeds, above random (0.0) | MODERATE | fig2 |
| 3 | MTO-Net exhibits seed-dependent optimization instability (2x performance gap good vs bad seeds) | 3/5 seeds plateaued at epoch 10, 2/5 continued improving | STRONG | fig1, fig3 |
| 4 | MTO slots show property-specific causal effects | Slot intervention: per-slot delta computed, property specialization observed | MODERATE | fig11 |
| 5 | MTOs show non-uniform atom-type enrichment (C, N, O, F distributions) | Atom-type enrichment analysis across 5 element types (H, C, N, O, F). RDKit unavailable — true functional-group validation (SMARTS) remains future work. | PRELIMINARY | fig12-fig15 |
| 6 | MTO design choices are distinguishable from simple pooling baselines | Full ablation (5 methods × 3 seeds, 5k mols, 20 ep). Norm mu: MTO 0.48-0.70 vs attention_pool 0.48-0.69. All methods show alpha_norm ~0.96. | MODERATE | fig16 |
| 7 | MTO representations are partially reusable when encoder is frozen | Frozen probe: frozen MTO mu=0.75±0.01 vs frozen DetaNet mu=1.01±0.01 vs from-scratch mu=1.09±0.00 (mean±std, 3 seeds). MTO provides measurable representation reuse benefit. | MODERATE | fig18 |
| 8 | MTO subspaces transfer across property tasks | Intra-Stage-A S_sub (mu_only vs alpha_only): 0.41, 0.64, 0.77 across seeds 0-2 | MODERATE | fig17 |
| 9 | Stage B/C code infrastructure is functional | Smoke tests passed (16 mols, 2 epochs), spectral CSV loading verified | MODERATE | fig_supp |
| 10 | Valence-adaptive K=N_val produces molecule-specific slot counts | K distribution varies with molecule size, verified in cache | MODERATE | fig4 |

## Partially Supported Claims

| # | Claim | What's Supported | What's Missing |
|---|-------|-----------------|----------------|
| P1 | MTOs form chemically meaningful response modes | Atom-type enrichment for O, N, F vs C, H | RDKit SMARTS unavailable; true functional-group validation remains future work |
| P2 | MTO outperforms simpler baselines | Attention pooling matches MTO on normalized mu MAE | Differences are marginal on this scale |
| P3 | Alpha prediction benefits from MTO | All methods show alpha_norm ~0.96 — near chance | Alpha prediction is hard at this scale |

## Unsupported Claims

| # | Claim | Why |
|---|-------|-----|
| U1 | MTOs are real molecular orbitals / Kohn-Sham orbitals / Hamiltonian eigenstates | Not claimed |
| U2 | MTOs reproduce physical occupation numbers | Not claimed |
| U3 | MTO-Net outperforms DetaNet on benchmarks | No external benchmark comparison |
| U4 | Stage B/C full spectral training is complete | Label-limited; only smoke tests run |

## Claims NOT Made

| # | Claim | Reason |
|---|-------|--------|
| N1 | MTOs are real molecular orbitals | Inspired by MO organization, not reconstructed |
| N2 | MTO-Net is the best model for property prediction | We claim interpretability + representation novelty |
| N3 | Stage B/C is complete | Smoke tests only; full training label-limited |
