```text
**Verdict: BLOCK / REQUEST CHANGES**

The current artifacts are not scientifically auditable or publication-ready. Core model claims, several evaluation metrics, and Figures 16–18 are unsupported or invalid.

**Critical / Blocking**

1. **The model is not rotation-equivariant.** DetaNet exposes scalar `S` and tensor `T`, but only `S` reaches MTO assembly; ordinary MLPs emit Cartesian vector/tensor components. The rotation test suppresses failure with `assert True`. [detanet_adapter.py](/home/xiangyu_xie/MTO/src/mto/detanet_adapter.py:35), [mto_model.py](/home/xiangyu_xie/MTO/src/mto/mto_model.py:39), [mto_readout.py](/home/xiangyu_xie/MTO/src/mto/mto_readout.py:15), [test_translation_invariance.py](/home/xiangyu_xie/MTO/tests/test_translation_invariance.py:49)

2. **The claimed l-specific signed tensor assembly is not implemented.** Three signs are averaged into one scalar coefficient and applied only to invariant features; DetaNet tensor channels are discarded. [mto_module.py](/home/xiangyu_xie/MTO/src/mto/mto_module.py:127)

3. **Baseline, transfer, and frozen-probe metrics use incompatible normalization spaces.** Normalized predictions are compared directly with raw labels, then normalized a second time. The reported raw/normalized alpha ratio exactly follows `alpha_std`, confirming this error. Figures and rankings derived from these metrics are invalid. [run_baselines.py](/home/xiangyu_xie/MTO/scripts/eval/run_baselines.py:111), [reeval_ablation.py](/home/xiangyu_xie/MTO/scripts/eval/reeval_ablation.py:23), [run_stage_transfer.py](/home/xiangyu_xie/MTO/scripts/eval/run_stage_transfer.py:113), [run_frozen_probe.py](/home/xiangyu_xie/MTO/scripts/eval/run_frozen_probe.py:138)

4. **The full ablation is experimentally confounded.** “No-sign” zeroes coefficients instead of using positive attention; fixed-K, direct, and attention variants reuse and mutate the previously trained backbone. Training always uses split seed 0, while re-evaluation uses seed-specific splits, potentially evaluating on training molecules. [run_baselines.py](/home/xiangyu_xie/MTO/scripts/eval/run_baselines.py:220), [reeval_ablation.py](/home/xiangyu_xie/MTO/scripts/eval/reeval_ablation.py:72), [full_ablation_medium.slurm](/home/xiangyu_xie/MTO/jobs/full_ablation_medium.slurm:35)

5. **Stage transfer is not predictive transfer.** Single-task models have no held-out-task head, producing `NaN`; Figure 17 converts these missing values to zero error. Subspace similarity between independently trained models alone cannot establish transfer. [run_stage_transfer.py](/home/xiangyu_xie/MTO/scripts/eval/run_stage_transfer.py:200), [make_figures_16_18.py](/home/xiangyu_xie/MTO/scripts/figures/make_figures_16_18.py:87)

6. **The frozen-probe comparison is invalid and lacks provenance.** The direct baseline freezes its output `heads`, leaving random final layers while training only modules named `readout`. It probes Stage A tasks again, not Stage B/C reuse. No `frozen_probe_results.json` exists locally, although Figure 18 and positive claims exist. [run_frozen_probe.py](/home/xiangyu_xie/MTO/scripts/eval/run_frozen_probe.py:266), [FIGURE_MANIFEST.txt](/home/xiangyu_xie/MTO/outputs/figures/FIGURE_MANIFEST.txt:56), [21_claim_evidence_table.md](/home/xiangyu_xie/MTO/outputs/reports/21_claim_evidence_table.md:18)

7. **Stage B/C transfer is not wired.** The jobs omit `--init-from`; missing spectral labels can silently remove spectral losses while the job continues. Consequently, nominal Stage B/C runs may actually be from-scratch mu/alpha training. [train_stage.py](/home/xiangyu_xie/MTO/scripts/train_stage.py:282), [full_stage_b_transfer.slurm](/home/xiangyu_xie/MTO/jobs/full_stage_b_transfer.slurm:17), [training.py](/home/xiangyu_xie/MTO/src/mto/training.py:117)

8. **Final-figure source integrity fails.** Required Stage A CSVs, caches, checkpoints, and `outputs/figures/source_data/` are absent from this checkout. The bundle mixes final, preliminary, pending, and stale figures. It cannot be reproduced from the repository. [gen_stage_a_figures.py](/home/xiangyu_xie/MTO/scripts/gen_stage_a_figures.py:30), [19_final_figure_audit.md](/home/xiangyu_xie/MTO/outputs/reports/19_final_figure_audit.md:28)

**Major Issues**

1. **Stage A normalization leaks held-out data.** Statistics use the first 2,000 dataset records rather than the locked training split; evaluation also covers only the first 2,000 validation/test records despite “full” naming. Metrics are normalized component metrics, not physical-unit tensor metrics. [train_stage.py](/home/xiangyu_xie/MTO/scripts/train_stage.py:214), [eval_stage_a_full.py](/home/xiangyu_xie/MTO/scripts/eval_stage_a_full.py:22)

2. **The reported activity gate is not charge-conserving.** Main experiments use the unconstrained sigmoid gate, while reports describe a valence/charge-conserving mechanism available only in the optional Fermi–Dirac mode. [activity_gate.py](/home/xiangyu_xie/MTO/src/mto/activity_gate.py:90), [train_stage.py](/home/xiangyu_xie/MTO/scripts/train_stage.py:61)

3. **Checkpointing is insufficient for exact reproduction.** Checkpoints omit normalization, splits, model configuration, task definitions, data/version hashes, RNG and scheduler state. Resume restarts epoch scheduling, while `strict=False` loading permits incompatible encoders. [training.py](/home/xiangyu_xie/MTO/src/mto/training.py:140), [train_stage.py](/home/xiangyu_xie/MTO/scripts/train_stage.py:275)

4. **Functional-group claims are false as written.** The executed analysis explicitly uses atom-type proxies because RDKit is unavailable, while final reports claim SMARTS-based carbonyl/hydroxyl/aromatic enrichment. [functional_group_analysis.py](/home/xiangyu_xie/MTO/scripts/functional_group_analysis.py:40), [20_final_story.md](/home/xiangyu_xie/MTO/outputs/reports/20_final_story.md:46)

5. **Stage B/C limitation reports are comparatively honest, but their advertised commands are not executable as documented**, using unsupported stage names/options. The defensible evidence level is smoke-scale spectral plumbing, not completed physical-label validation. [17_stage_b_label_limitation.md](/home/xiangyu_xie/MTO/outputs/reports/17_stage_b_label_limitation.md:84), [18_stage_c_label_limitation.md](/home/xiangyu_xie/MTO/outputs/reports/18_stage_c_label_limitation.md:77)

6. **Slurm and CLI reproducibility is fragile.** Jobs commonly lack fail-fast handling; some pass unsupported arguments or incorrect checkpoint paths. The canonical evaluator has invalid imports and undefined names. [submit_ablation.sh](/home/xiangyu_xie/MTO/scripts/slurm/submit_ablation.sh:14), [eval_model.py](/home/xiangyu_xie/MTO/scripts/eval_model.py:15)

7. **Reports materially overclaim.** “Competitive,” “above random,” “transfer,” “essential,” “functional-group enrichment,” and frozen-reuse conclusions exceed the available controls and, in several cases, rely on invalid calculations. [21_claim_evidence_table.md](/home/xiangyu_xie/MTO/outputs/reports/21_claim_evidence_table.md:12), [20_final_story.md](/home/xiangyu_xie/MTO/outputs/reports/20_final_story.md:74)

**Minor Suggestions**

- `K=N_val` and main masking are internally consistent for neutral QM9S molecules, but remain a neutral-atom heuristic with no charge/radical handling and no guard for `K > 128`. [valence.py](/home/xiangyu_xie/MTO/src/mto/valence.py:3), [mto_module.py](/home/xiangyu_xie/MTO/src/mto/mto_module.py:25)
- Strengthen mask tests: the “padded slots zero” test checks only mask flags, and padding invariance compares identical inputs. [test_mto_mask.py](/home/xiangyu_xie/MTO/tests/test_mto_mask.py:41)
- Replace the assumed random subspace baseline of `0.0` with an empirical, dimension-matched null distribution.
- Validate all documented Python entry points in CI.

**Verification**

`pytest -q`: **75 passed, 19 skipped**.  
`compileall`: failed in three advertised scripts, including `run_all_medium.py` and `run_all_smoke.py`.  
Shell syntax checks passed. Independent code-review and architecture reviews both concluded **REQUEST CHANGES / BLOCK**. No files were modified.


Reading additional input from stdin...
