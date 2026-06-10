# Final Figure Audit Report

**Report**: 19_final_figure_audit.md
**Date**: 2026-06-10
**Branch**: tensor-mto-v2

---

## Final Figure Summary

| # | Figure | Status | Source | Script | Scientific Question |
|---|--------|--------|--------|--------|-------------------|
| 1 | fig1_seed_performance | FINAL | Stage A metrics | gen_stage_a_figures.py | Does MTO-Net predict mu+alpha? |
| 2 | fig2_seed_subspace_stability | FINAL | Stage A analysis | gen_stage_a_figures.py | Are MTO subspaces stable across seeds? |
| 3 | fig3_training_dynamics | FINAL | Stage A logs | gen_stage_a_figures.py | How does training converge? |
| 4 | fig4_mto_cache_overview | FINAL | MTO caches | gen_stage_a_figures.py | What do MTOs look like? |
| 5 | fig5_good_bad_mto_comparison | FINAL | MTO caches | gen_stage_a_figures.py | How do good/bad seeds differ? |
| 6 | fig6_property_specific_mto_usage | FINAL | MTO caches | gen_stage_a_figures.py | Which MTO slots serve which property? |
| 7 | fig7_mu_alpha_specialization | FINAL | MTO caches | gen_stage_a_figures.py | How specialized are MTOs? |
| 8 | fig8_representative_mto_atom_maps | FINAL | MTO caches | gen_stage_a_figures.py | Where do MTOs localize on molecules? |
| 9 | fig9_good_seed_atom_map_consistency | FINAL | MTO caches | gen_stage_a_figures.py | Are atom maps consistent within good seeds? |
| 10 | fig10_good_bad_atom_map_comparison | FINAL | MTO caches | gen_stage_a_figures.py | How do good/bad atom maps differ? |
| 11 | fig11_representative_slot_intervention | FINAL | Intervention analysis | interpretability_analysis.py | Do MTO slots causally affect predictions? |
| 12 | fig12_functional_group_enrichment_overview | FINAL | FG analysis | functional_group_analysis.py | Do MTOs enrich for functional groups? |
| 13 | fig13_mu_alpha_functional_group_association | FINAL | FG analysis | functional_group_analysis.py | Are associations property-specific? |
| 14 | fig14_representative_functional_group_maps | FINAL | FG analysis | functional_group_analysis.py | Where do FG-enriched MTOs localize? |
| 15 | fig15_good_bad_functional_group_enrichment | FINAL | FG analysis | functional_group_analysis.py | Is enrichment stronger in good seeds? |
| 16 | fig16_baseline_ablation_summary | **FINAL** | Ablation re-eval (86512) | make_figures_16_18.py | Do MTO design choices matter? |
| 17 | fig17_stage_transfer_stability | **FINAL** | Stage transfer re-eval (86517) | make_figures_16_18.py | Do MTO subspaces transfer? |
| 18 | fig18_frozen_probe_reuse | **PENDING** | Frozen probe (86515 running) | make_figures_16_18.py | Are MTO representations reusable? |
| S | fig_supp_stage_b_spectral_smoke | PRELIMINARY | Stage B smoke | train_stage.py | Does Stage B code work? |

## Regenerated Figures (2026-06-10)

- **fig16**: Regenerated from real data — 5 methods × 3 seeds × 20 epochs. Both raw and normalized MAE shown. Normalized alpha ~0.96 for all methods (near-chance).
- **fig17**: Regenerated from real data — 3 conditions × 3 seeds. S_sub mu_only vs alpha_only = [0.41, 0.64, 0.77]. Partial above-random transfer.
- **fig18**: Pending frozen probe completion (job 86515).

## Figure Checklist

Each figure has:
- [x] Source data
- [x] Generating script
- [x] Caption
- [x] Scientific question
- [x] Status: final / preliminary / label-limited / failed
