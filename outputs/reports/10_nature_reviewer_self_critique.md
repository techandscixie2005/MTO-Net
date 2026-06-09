# Pre-Submission Self Critique — MTO-Net

**Audience:** Nature Computational Science
**Reviewer perspective:** AI for Chemistry, Equivariant ML, Molecular Property Prediction

## Claim 1: MTOs form stable molecule-level response modes

**Evidence strength: Moderate**
- Seed subspace stability S_sub ~0.51 is above random but far from perfect
- 3/5 seeds plateau early — training instability may undermine stability claims
- Only 5 seeds evaluated — insufficient for robust variance estimation
- Top-r=5 heuristic choice is essentially arbitrary and could bias S_sub

**Risk:** Reviewer will ask: "Is S_sub=0.51 meaningfully different from any attention pooling baseline? Where's the comparison?"

**Fix:** Report baseline S_sub values for direct readout and attention pooling to establish the null distribution.

## Claim 2: MTOs show chemically meaningful localization

**Evidence strength: Weak-to-moderate**
- Functional group enrichment is statistically significant for polar groups
- But enrichment is correlational, not causal
- No comparison against random baseline or chemical null models
- SMARTS coverage limited to ~10 groups
- Could be an artifact of elemental composition rather than functional group identity

**Risk:** Reviewer will note that electronegative atoms (O, N, F) naturally contribute to dipole, so "carbonyl enrichment" could just be "oxygen enrichment."

**Fix:** Include atom-type-only enrichment as explicit negative control. Show that functional-group enrichment exceeds atom-type baseline.

## Claim 3: MTOs support property-specific response organization

**Evidence strength: Moderate**
- Slot intervention shows property specialization (some slots more important for mu, others for alpha)
- But specialization is partial, not crisp — many slots contribute to both
- Only 2 properties tested (mu + alpha). Need IR/Raman/UV to demonstrate true multi-property organization
- No baseline comparison: attention pooling might show similar per-k specialization

**Fix:** Show attention pooling + fixed-K MTO intervention results for comparison.

## Claim 4: Valence-adaptive slot count improves over fixed-K

**Evidence strength: Not yet demonstrated**
- Fixed-K ablation not run
- No systematic comparison of K=N_val vs K=10, 20, 30, 50
- Larger K may simply provide more parameters, and improvements may be from capacity, not adaptivity

**Fix:** Run fixed-K ablation. If improvements are marginal, be honest about this.

## Claim 5: Signed coefficient provides meaningful bonding/antibonding organization

**Evidence strength: Not yet demonstrated**
- No-sign MTO ablation not run
- Claim rests on architecture design, not experimental evidence
- Without visualization of sign patterns for representative molecules (e.g., ethylene, benzene), the claim is purely architectural

**Fix:** Run no-sign ablation. Visualize c_ki signs for representative molecules.

## Major Concerns

### 1. Missing baseline comparisons
Without direct readout, attention pooling, and no-sign MTO baselines, all stability/interpretability claims rest on the MTO's internal properties, not on controlled comparison.

### 2. Seed instability
3/5 bad seeds suggest optimization problems. If the method cannot reliably reach good minima, it's not practically useful regardless of interpretability.

### 3. Evidence chain is incomplete
- Stage A only (mu+alpha)
- No full Stage B/C training
- No frozen probe experiment
- No comparison against literature baselines

### 4. Scientific scope vs QM9S limitations
QM9S provides only mu and alpha. Without Hessian/normal modes, the physical-quantity-to-spectrum route is unavailable. Direct spectrum regression from CSVs is a fallback, not the preferred approach.

## Recommendations for Revision

1. Run all Tier-1 baselines before submission
2. Report S_sub for all methods (not just MTO)
3. Include negative controls for functional group analysis
4. Either complete Stage B/C or explicitly scope the paper to mu+alpha
5. Retry bad seeds with different optimization strategies
6. Add sign visualization for representative molecules
7. Be explicit about QM9S label limitations in the main text
8. Do not claim "molecular orbital" without strong chemical evidence

## Overall Assessment

**Current state:** Promising but incomplete. The core idea (signed, center-free, valence-adaptive MTO assembly) is novel and sound. The initial results (seed stability, chemical enrichment, slot specialization) are encouraging. However, missing baselines and incomplete training prevent firm conclusions.

**Readiness for submission:** NOT YET. Need at minimum:
- Tier-1 baselines run and compared
- More robust seed stability analysis
- Sign pattern visualization
- Honest discussion of limitations

**Risk level if submitted now:** HIGH risk of rejection with "interesting idea but insufficient evidence" reviews.
