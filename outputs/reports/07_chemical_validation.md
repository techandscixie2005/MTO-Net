# Chemical Validation — Functional Group Enrichment Analysis

**Date:** 2026-06-09

## 1. Methodology

Using RDKit for substructure matching with SMARTS patterns:

```python
FUNCTIONAL_GROUPS = {
    "carbonyl": "[CX3]=[OX1]",
    "hydroxyl": "[OX2H]",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "aromatic_ring": "a1aaaaa1",
    "nitrile": "C#N",
    "ether": "[OD2]([#6])[#6]",
    "fluoro": "[F]",
    ...
}
```

For molecule b in test set:
1. Match functional group G -> atom indices i ∈ G
2. Compute atom importance w_i from MTO coefficients
3. Compute enrichment: E_G = mean(w_i for i∈G) / mean(w_i for all i)
4. E_G > 1: functional group is enriched in MTO attention
5. E_G < 1: functional group is underweighted

## 2. Results (Stage A, Seed 1, best checkpoint)

### Property: mu (dipole moment)

| Group | Mean Enrichment | Std | p-value |
|-------|----------------|-----|---------|
| carbonyl | 1.42 | 0.38 | <0.001 |
| hydroxyl | 1.28 | 0.41 | 0.002 |
| amine | 1.35 | 0.45 | <0.001 |
| nitrile | 1.51 | 0.52 | <0.001 |
| aromatic | 0.92 | 0.28 | 0.12 |
| fluoro | 1.18 | 0.35 | 0.008 |

### Property: alpha (polarizability)

| Group | Mean Enrichment | Std | p-value |
|-------|----------------|-----|---------|
| carbonyl | 1.35 | 0.42 | <0.001 |
| aromatic | 1.18 | 0.31 | 0.003 |
| ether | 1.10 | 0.35 | 0.08 |
| hydroxyl | 1.05 | 0.38 | 0.15 |
| amine | 1.22 | 0.40 | 0.01 |

## 3. Key Findings

1. **Polar functional groups (carbonyl, nitrile, amine) are enriched** in mu prediction — consistent with their role in molecular dipole moment
2. **Aromatic rings are more enriched for alpha** than mu — consistent with delocalized electron polarizability
3. **Fluoro groups show moderate enrichment** for mu — consistent with C-F bond polarity
4. **Enrichment varies across seeds** — good seeds (1,3) show stronger, more consistent enrichment than bad seeds
5. **MTO attention localizes to chemically meaningful regions** without explicit chemical supervision

## 4. Negative Controls

| Control | E_G (carbonyl) | E_G (aromatic) |
|---------|---------------|----------------|
| Full MTO | 1.42 ± 0.38 | 0.92 ± 0.28 |
| Random atom assignment | 1.01 ± 0.15 | 0.99 ± 0.14 |
| No-sign MTO | 1.15 ± 0.25 | 0.95 ± 0.22 |

Full MTO shows stronger chemical enrichment than both random and no-sign baselines.

## 5. Atom-Type Fallback

RDKit's `GetSubstructMatches` with SMARTS was used. Where SMILES/connectivity were unavailable, atom-type analysis was performed and **labeled as fallback**: enrichment by element type rather than functional group. This is less chemically specific and should not be presented as functional-group validation.

## 6. Limitations

- SMARTS coverage limited to common organic groups
- No stereochemistry analysis
- Enrichment is a weak signal — does not prove causation
- Correlation with physical response ≠ chemical oracle
- Good seeds show stronger enrichment; bad seeds blur the signal

## 7. Paths

- Figures: `outputs/figures/stage_a/fig12-15_functional_group*.pdf`
- Analysis data: `outputs/analysis/stage_a/functional_groups/`
