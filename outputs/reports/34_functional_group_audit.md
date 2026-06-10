# Functional-Group Audit Report

**Report**: 34_functional_group_audit.md  
**Date**: 2026-06-10  
**Auditor**: Ralph polishing session

---

## Classification

**Case B: Atom-type proxy only. No RDKit SMARTS functional-group analysis exists.**

## Evidence

From `scripts/functional_group_analysis.py`:

1. **Line 2**: `"""Stage A functional-group enrichment analysis using atom-type categories (RDKit unavailable)."""`
2. **Line 40**: `print(f"RDKit: NOT INSTALLED — using atom-type enrichment")`
3. **Line 31**: `ATOM_TYPES = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}` — atomic numbers, not functional groups
4. **Lines 54-63**: "Atom-type groups" are defined by atomic number ranges:
   - `carbon_C`: z=[6] → "Carbon" (actually: any carbon atom)
   - `oxygen_O`: z=[8] → "Oxygen" (actually: any oxygen atom)
   - `heteroatom`: z=[7,8,9] → "Heteroatom (N,O,F)"
   - `heavy_atom`: z=[6,7,8,9] → "Heavy atom (C,N,O,F)"
5. **No imports of rdkit, Chem, MolFromSmiles, or GetSubstructMatches**
6. **No SMARTS patterns anywhere in the codebase**
7. **Report lines 47, 69, 460, 475, 483, 501, 503, 510, 517** all explicitly state "RDKit unavailable" or "atom-type proxy"

## What This Means

- "Functional-group enrichment" in the current figures (fig12–fig15) is actually **atom-type enrichment**: which elements do high-contribution MTO atoms tend to be?
- Claims about "carbonyl enrichment" or "hydroxyl enrichment" are **unsupported** because the analysis cannot distinguish a carbonyl carbon from an alkane carbon — both are `z=6`
- The figures are correctly labeled as "atom-type enrichment" in the figure captions (the generating script annotates this), but the figure **filenames** contain "functional_group"

## Required Changes

### Figure filenames unchanged (for continuity) but captions updated

The generating script already annotates figures with "RDKit unavailable — using atom-type proxy". The filenames `fig12_functional_group_enrichment_overview.pdf` etc. are legacy. **Recommendation for paper**: use "atom-type enrichment" in figure captions.

### Report wording changes

| File | Old Wording | New Wording |
|------|------------|-------------|
| 21_claim_evidence_table.md | "functional group enrichment (carbonyl, hydroxyl, aromatic)" | "atom-type enrichment (C, N, O, F distributions)" |
| 22_limitations.md | Add limitation about FG analysis | Already present but strengthen |
| 20_final_story.md | If mentions "functional group" | "atom-type enrichment" |
| 24_nature_reviewer_self_critique.md | If mentions "functional group" | "atom-type enrichment" |

### Paper wording recommendation

Replace:
> "MTOs show measurable enrichment for polar functional groups (carbonyl, hydroxyl, aromatic)"

With:
> "MTO-contributing atoms show non-uniform distribution across atom types (C, N, O, F), consistent with the hypothesis that MTOs learn chemically-relevant response modes. Functional-group-level validation via RDKit SMARTS remains future work."

## Conclusion

- **RDKit**: Not used
- **SMARTS**: Not used
- **Connectivity**: Not analyzed
- **Result type**: Atom-type proxy enrichment
- **Scientific claim level**: Preliminary / exploratory — shows chemical-awareness at element level, not functional-group level
- **Recommended**: Downgrade all "functional-group" claims to "atom-type enrichment"; add explicit future-work note for RDKit SMARTS analysis
