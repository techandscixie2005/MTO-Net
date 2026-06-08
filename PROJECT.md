# PROJECT.md

# MTO-Net Project Implementation Blueprint

## 0. Project Identity

Project name:

```text
MTO-Net: Molecular Tensor Orbital Network
```

Full research title:

```text
Valence-Adaptive Molecular Tensor Orbitals for Equivariant Molecular Response Learning
```

Chinese title:

```text
面向分子响应性质学习的价电子自适应分子张量轨道网络
```

This project implements MTO-Net on top of the DetaNet source code.

DetaNet is used as the atomic-level equivariant tensor representation backbone. MTO-Net does not rewrite DetaNet's input pipeline, molecular graph construction, message passing layers, coordinate handling, tensor feature update, or atom-level representation learning. Instead, MTO-Net adds a molecule-level representation module after DetaNet's updated atom-level tensor representations.

The new module constructs a set of molecule-level, orbital-like, equivariant response modes called:

```text
Molecular Tensor Orbitals, MTOs
```

The central goal is not merely to outperform DetaNet as a black-box predictor. The central goal is to test whether updated atom-level equivariant tensor fields can be reorganized into stable, chemically meaningful, physically interpretable, and reusable molecule-level response modes.

---

## 1. Core Scientific Question

The core scientific question is:

```text
Can an equivariant molecular neural network form stable, reusable, chemically meaningful molecule-level response modes before property readout?
```

In mathematical form, conventional equivariant molecular models are usually organized as:

[
{Z_i,\mathbf r_i}
\rightarrow
{h_i^{(l)}}
\rightarrow
\mathrm{pooling/readout}
\rightarrow
y
]

MTO-Net changes the middle representation into:

[
{Z_i,\mathbf r_i}
\rightarrow
{h_i^{(l)}}
\rightarrow
{\mathcal O_k^{(l)}}_{k=1}^{K}
\rightarrow
\text{physical response quantities}
\rightarrow
\text{spectra}
]

where:

[
\mathcal O_k
]

is the (k)-th Molecular Tensor Orbital.

MTOs are not physical Kohn-Sham orbitals, not quantum-chemical molecular orbitals, not wavefunctions, and not Hamiltonian eigenstates. They are molecule-level equivariant latent response modes inspired by molecular-orbital organization.

Safe definition:

```text
MTOs are molecule-level equivariant latent response modes inspired by molecular orbital organization, rather than physical molecular orbitals.
```

Chinese definition:

```text
MTO 是等变隐空间中的分子级轨道式响应模式，而不是真实 Kohn-Sham 轨道或量子化学波函数。
```

---

## 2. Research Boundary

This first implementation is deliberately constrained.

The project does not aim to:

1. reconstruct quantum-chemical wavefunctions;
2. predict physical molecular orbitals;
3. predict Kohn-Sham orbitals;
4. predict Hamiltonians;
5. perform self-consistent field iterations;
6. perform transition-state search;
7. perform molecular dynamics;
8. build a universal quantum chemistry engine;
9. model charged molecular states;
10. model spin;
11. predict explicit electronic occupations;
12. claim that each MTO corresponds to a real occupied or virtual orbital.

The project does aim to:

1. construct MTOs from DetaNet atom-level tensor representations;
2. set the number of MTO slots using the total valence electron count;
3. train MTOs to predict molecular response quantities;
4. generate spectra from predicted physical response quantities whenever possible;
5. evaluate seed stability at the MTO subspace level;
6. evaluate chemical meaning through functional-group and fragment-level analysis;
7. evaluate physical meaning through response-property readouts and intervention;
8. evaluate reusability through stage-to-stage transfer and frozen/readout-probe experiments.

---

## 3. Key Clarification: Charge vs Electron Count

The MVP does not model molecular charge states.

This does not mean electron-count information is ignored.

All molecules are treated as neutral. The total molecular valence electron count is used as a capacity prior for the MTO bank.

[
K = N_{\mathrm{val}} = \sum_i v(Z_i)
]

where (v(Z_i)) is the valence electron count of atom (i).

This means:

```text
No charge-state modeling.
No formal-charge correction.
No Q term.
No ion-specific handling.
But K is explicitly set to the molecular total valence electron count.
```

The value of (K) is a valence-adaptive capacity prior. It is not a claim that each MTO is a real molecular orbital or that each MTO has a true physical occupation number.

For QM9S, the expected primary elements are:

```python
VALENCE_ELECTRONS = {
    1: 1,   # H
    6: 4,   # C
    7: 5,   # N
    8: 6,   # O
    9: 7,   # F
}
```

During Phase 0 audit, Claude Code must inspect the actual dataset elements. If elements outside this table appear, the script must stop and report them instead of silently assigning valence 0.

MVP config:

```yaml
mto:
  k_mode: "total_valence_electrons"
  charge_mode: "neutral_only"
  use_formal_charge: false
  k_equals: "N_val"
  k_max_cap: null
```

---

## 4. Primary Scientific Claim

The primary claim of this project is not:

```text
MTO-Net is always more accurate than DetaNet.
```

The primary claim is:

```text
MTO-Net forms stable, reusable, physically meaningful molecule-level response modes from atom-level equivariant tensor fields.
```

Prediction accuracy is necessary as a sanity check. However, the main scientific evidence should come from:

1. prediction sanity;
2. seed subspace stability;
3. stage-to-stage stability;
4. physical-response consistency;
5. functional-group enrichment;
6. slot intervention;
7. frozen/readout transferability;
8. comparison with pooling/token baselines.

Performance better than DetaNet is welcome, but not required for the first proof-of-concept paper.

---

## 5. Overall Model Architecture

The model pipeline is:

```text
Input molecular structure
  ↓
DetaNet atom-level equivariant tensor encoder
  ↓
Updated atom-level tensor representations
  ↓
DetaNet-to-MTO adapter
  ↓
Valence-adaptive MTO bank, K = N_val
  ↓
Signed center-free MTO assembly
  ↓
MTO activity gate
  ↓
Physical response readouts
  ↓
Spectrum construction from physical quantities
  ↓
Interpretability, stability, and chemical validation
```

Mathematically:

[
{Z_i,\mathbf r_i}_{i=1}^N
\rightarrow
h_i
]

[
h_i
\rightarrow
\mathcal O_k,\quad k=1,\ldots,K=N_{\mathrm{val}}
]

[
{\mathcal O_k}
\rightarrow
\hat \mu,\hat \alpha,\widehat{H},\widehat{\partial\mu},\widehat{\partial\alpha},\widehat{IR},\widehat{Raman},\widehat{UV}
]

The exact available targets must be determined by local QM9S dataset audit. Do not assume all physical quantities exist before inspecting the actual files.

---

## 6. Core Implementation Strategy

Because the MTO module itself is not expected to be highly complex, Claude Code should not implement only one tiny component at a time.

The correct execution strategy is:

```text
Phase 0 audit
  ↓
Complete full project codebase
  ↓
Full smoke test
  ↓
Small/medium sanity run
  ↓
Full QM9S multi-seed run
  ↓
Analysis, visualization, and reports
```

After Phase 0 audit, Claude Code should implement the complete project code path before running smoke tests.

The complete codebase must include:

1. DetaNet adapter and atom-level representation hook;
2. valence-electron counter and (K=N_{\mathrm{val}}) MTO bank;
3. signed MTO assembly;
4. MTO activity gate;
5. physical response readouts;
6. spectrum construction from predicted physical quantities whenever labels and formulas are available;
7. Stage A/B/C training scripts;
8. evaluation scripts;
9. MTO cache scripts;
10. seed subspace stability analysis;
11. stage-to-stage stability analysis;
12. functional-group chemical validation;
13. slot intervention analysis;
14. frozen/readout-probe analysis;
15. plotting scripts;
16. Nature-style figure generation scripts;
17. Slurm scripts for H200/H100/A800 after resource detection;
18. final report generator.

Only after the full code path exists should Claude Code run smoke tests.

---

## 7. DetaNet Integration Rule

Do not rewrite DetaNet.

DetaNet is used as:

```text
z, pos, batch
  → DetaNet backbone
  → updated atom-level tensor representation
```

MTO-Net starts after the updated atom-level representation.

The key implementation file is:

```text
mtonet/models/detanet_adapter.py
```

The adapter must:

1. import or wrap the existing DetaNet model;
2. expose updated atom-level representations;
3. preserve original DetaNet forward behavior when MTO is disabled;
4. record tensor shapes during audit;
5. support `return_atom_features=True`;
6. avoid changing DetaNet internals unless strictly necessary.

Important rule:

```text
Do not assume DetaNet hidden features are explicitly stored as l=0,1,2 dictionaries.
```

Phase 0 must inspect the actual source code and runtime tensors.

If DetaNet exposes a flat hidden representation, the adapter should operate on the flat representation first. Later decomposition into tensor orders can be added only if the representation format is confirmed.

---

## 8. MTO Module Design

### 8.1 Input

The MTO module receives atom-level representations:

[
h_i
]

from the DetaNet adapter.

The representation may be:

```text
Tensor[N_atoms, C]
```

or

```text
dict[l -> Tensor[N_atoms, C_l, 2l+1]]
```

depending on the actual DetaNet code.

The implementation must support the confirmed DetaNet format.

---

### 8.2 MTO Slot Count

For each molecule (b):

[
K_b = N_{\mathrm{val},b}
]

Batch-level padding:

[
K_{\max} = \max_b K_b
]

Mask:

```python
mto_mask[b, k] = (k < K_b)
```

All readouts, losses, cache, and plots must respect `mto_mask`.

---

### 8.3 Routing Score

For molecule (b), atom (i), and MTO slot (k):

[
e_{ki} = \mathrm{MLP}_r(s_i,q_k)
]

where:

* (s_i) is an invariant summary of atom representation (h_i);
* (q_k) is the learnable slot embedding;
* (e_{ki}) is scalar;
* no absolute coordinate is used;
* routing must be center-free.

---

### 8.4 Attention Coefficient

For each MTO slot (k), atom coefficients are normalized over atoms in the same molecule:

[
a_{ki} =
\frac{\exp(e_{ki})}
{\sum_{j \in b}\exp(e_{kj})}
]

---

### 8.5 Signed Coefficient

MTO assembly must allow signed combinations, inspired by bonding/antibonding organization.

[
s_{ki} = \tanh(\mathrm{MLP}_s(s_i,q_k))
]

[
c_{ki}
======

\frac{a_{ki}s_{ki}}
{\sum_{j\in b}|a_{kj}s_{kj}|+\epsilon}
]

This gives a signed, normalized, center-free atom-to-MTO coefficient.

---

### 8.6 MTO Construction

If the DetaNet representation is flat:

[
\mathcal O_k
============

\sum_{i\in b}c_{ki}h_i
]

If the DetaNet representation is explicitly split by tensor order:

[
\mathcal O_k^{(l)}
==================

\sum_{i\in b}c_{ki}h_i^{(l)}
]

The coefficient (c_{ki}) is scalar, so it preserves the transformation type of the tensor representation.

---

### 8.7 Activity Gate

Each MTO slot receives a learned activity score:

[
g_k = \sigma(\mathrm{MLP}_g(\mathcal O_k))
]

The active MTO is:

[
\tilde{\mathcal O}_k = g_k\mathcal O_k
]

Interpretation:

```text
g_k measures how active an MTO slot is for molecular response prediction.
It is not a strict physical occupation number in the MVP.
```

The activity gate may be regularized lightly, but should not dominate prediction loss.

---

## 9. Physical Quantity First, Spectrum Second

The project must prioritize physical response quantities before spectral curves.

The preferred path is:

```text
MTOs
  → physical response quantities
  → spectral intensities
  → broadened spectra
```

For IR and Raman, the preferred route is:

```text
MTO bank
  → Hessian or vibrational quantities
  → normal modes / frequencies
  → dipole derivatives
  → polarizability derivatives
  → IR intensities
  → Raman intensities
  → broadened spectra
```

Direct spectrum regression is allowed only as:

1. an auxiliary loss;
2. a fallback if physical labels are unavailable;
3. a debugging baseline.

It should not be the main scientific route if the required physical quantities are available in QM9S.

During Phase 0, Claude Code must inspect the local QM9S dataset and report which of the following labels exist:

```text
dipole moment
polarizability tensor
Hessian
vibrational frequencies
normal modes
dipole derivatives
polarizability derivatives
IR intensities
Raman intensities
broadened IR spectrum
broadened Raman spectrum
UV transition energies
oscillator strengths
transition dipoles
broadened UV spectrum
NMR shielding
```

The training code must support automatic enabling/disabling of tasks based on available labels.

Config design:

```yaml
tasks:
  mu: true
  alpha: true
  hessian: auto
  frequencies: auto
  normal_modes: auto
  dipole_derivative: auto
  polar_derivative: auto
  ir_intensity: auto
  raman_intensity: auto
  ir_spectrum: auto
  raman_spectrum: auto
  uv_transition: auto
  uv_spectrum: auto
```

`auto` means:

```text
enable if labels exist;
otherwise skip and record the reason in the audit report.
```

Never fabricate missing physical labels.

---

## 10. Stage Design

### 10.1 Stage A: Fundamental Molecular Response

Tasks:

[
\mathcal T_A = {\mu,\alpha}
]

Purpose:

1. train the simplest physically meaningful response model;
2. verify MTO formation;
3. evaluate seed stability;
4. visualize MTO atom maps;
5. perform initial chemical validation.

Stage A is the foundation of the entire project.

If Stage A MTOs are not stable, do not over-interpret Stage B/C.

---

### 10.2 Stage B: Vibrational Response and IR/Raman

Preferred tasks:

[
\mathcal T_B =
{\mu,\alpha,H,\partial\mu,\partial\alpha,IR,Raman}
]

Exact enabled tasks depend on available QM9S labels.

The preferred path is:

```text
MTOs
  → Hessian or vibrational quantities
  → dipole derivatives
  → polarizability derivatives
  → IR/Raman intensities
  → IR/Raman spectra
```

Stage B must initialize from Stage A checkpoint.

Stage B should test whether MTOs learned from (\mu,\alpha) remain stable after adding vibrational response tasks.

---

### 10.3 Stage C: Electronic Response and UV

Preferred tasks:

```text
mu
alpha
vibrational response quantities
IR/Raman
UV-related response quantities
UV spectrum
```

For UV, the preferred route is:

```text
MTOs
  → electronic response quantities
  → UV spectrum
```

If transition-level labels are unavailable or unstable, broadened UV spectrum prediction can be used as a response-level supervision target.

Stage C must initialize from Stage B checkpoint.

Stage C tests whether MTOs remain reusable after introducing a more delocalized electronic-response task.

---

## 11. Training Philosophy

Prediction performance is a sanity check.

The main evidence must come from:

1. MTO stability across seeds;
2. MTO stability across stages;
3. MTO chemical localization;
4. MTO physical intervention effects;
5. MTO reuse in frozen/readout-probe settings.

The model should be trained to a reasonable predictive quality before interpretability analysis. However, do not delay all analysis until perfect prediction metrics are achieved.

---

## 12. Required Project Structure

If the repository already has a structure, preserve it as much as possible. Add the following files and folders only when needed.

Recommended final structure:

```text
MTO-Net/
  TOTAL.md
  PROJECT.md
  CLAUDE.md

  detanet_model/
    detanet.py
    modules/
    ...

  mtonet/
    __init__.py

    models/
      detanet_adapter.py
      mtonet.py

    modules/
      valence.py
      mto_assembly.py
      activity_gate.py
      readouts.py
      response_heads.py
      spectrum_builder.py
      tensor_utils.py

    losses/
      multitask_losses.py
      mto_losses.py

    analysis/
      mto_cache.py
      subspace_similarity.py
      seed_stability.py
      stage_transfer.py
      slot_intervention.py
      functional_groups.py
      chemical_validation.py
      frozen_probe.py

    plotting/
      nature_style.py
      plot_mto_heatmap.py
      plot_seed_stability.py
      plot_stage_transfer.py
      plot_functional_group_enrichment.py
      plot_intervention.py
      plot_ablation.py

  scripts/
    audit/
      00_env_check.sh
      01_repo_check.sh
      02_dataset_audit.py
      03_detanet_forward_check.py
      04_detect_slurm_gpu.sh

    train/
      train_stage_a.py
      train_stage_b.py
      train_stage_c.py
      train_baseline.py
      train_ablation.py
      train_frozen_probe.py

    eval/
      eval_prediction.py
      eval_mto_cache.py
      eval_seed_stability.py
      eval_stage_transfer.py
      eval_chemical_validation.py
      eval_intervention.py
      eval_frozen_probe.py

    slurm/
      generate_slurm_templates.py
      submit_smoke.sh
      submit_stage_a_array.sh
      submit_stage_b_array.sh
      submit_stage_c_array.sh
      submit_ablation.sh

    figures/
      make_all_debug_figures.py
      make_all_final_figures.py

  configs/
    base.yaml
    mtonet_stage_a.yaml
    mtonet_stage_b.yaml
    mtonet_stage_c.yaml
    detanet_baseline.yaml
    pooling_baseline.yaml
    ablation_no_sign.yaml
    ablation_fixed_k.yaml
    frozen_probe.yaml
    plotting.yaml

  outputs/
    logs/
    reports/
    checkpoints/
    metrics/
    cache/
    figures/
      debug/
      final/
      source_data/
    splits/

  tests/
    test_valence.py
    test_mto_shapes.py
    test_mto_mask.py
    test_forward_smoke.py
    test_checkpoint_reload.py
    test_translation_invariance.py
    test_rotation_equivariance.py
    test_permutation_invariance.py
    test_padding_mask_invariance.py
    test_subspace_similarity.py
    test_cache_schema.py
```

---

## 13. Mandatory Tests

Before full QM9S training, the following tests must pass.

### 13.1 Shape and Mask Tests

```text
test_valence.py
test_mto_shapes.py
test_mto_mask.py
test_forward_smoke.py
```

Required checks:

1. (K=N_{\mathrm{val}}) is correct;
2. `mto_mask` is correct;
3. padded slots do not affect output;
4. MTO tensors have correct shape;
5. forward pass works on a small batch.

---

### 13.2 Training and Checkpoint Tests

```text
test_checkpoint_reload.py
```

Required checks:

1. model can train for at least one step;
2. loss is finite;
3. checkpoint can be saved;
4. checkpoint can be reloaded;
5. reloaded model gives identical or near-identical output.

---

### 13.3 Equivariance and Invariance Tests

```text
test_translation_invariance.py
test_rotation_equivariance.py
test_permutation_invariance.py
test_padding_mask_invariance.py
```

Minimum checks:

1. translating all coordinates should not change invariant predictions;
2. rotating input should rotate vector outputs such as (\mu);
3. scalar spectra should remain invariant under rotation;
4. permuting atom order should not change molecule-level predictions;
5. padding should not affect valid MTO slots or outputs.

These tests are mandatory because MTO assembly must not break the equivariance inherited from DetaNet.

---

## 14. MTO Cache Design

Evaluation must save MTO interpretability cache.

However, cache must be size-controlled.

Default cache config:

```yaml
cache:
  save_train_cache: false
  save_valid_cache: true
  save_test_cache: true
  save_full_cache_epochs: ["best", "last"]
  max_molecules_per_epoch: 512
  save_selected_molecules: true
  compression: "npz_compressed"
```

Each cached molecule should include:

```python
{
    "mol_id": ...,
    "smiles": ...,
    "z": ...,
    "pos": ...,

    "K": ...,
    "mto_mask": ...,

    "routing_logits": ...,
    "routing_attention": ...,
    "signed_coeff": ...,
    "activity": ...,

    "mto_tensor": ...,
    "atom_tensor_norm": ...,

    "pred": ...,
    "target": ...,

    "seed": ...,
    "stage": ...,
    "checkpoint": ...,
    "git_commit": ...,
}
```

Do not save massive full-dataset cache at every epoch.

---

## 15. MTO Atom Maps

For MTO (k) and atom (i), define atom contribution:

If using flat representation:

[
w_{ki}^{MTO}
============

\left|
c_{ki}h_i
\right|^2
]

If using tensor-order split representation:

[
w_{ki}^{MTO}
============

\sum_l
\left|
c_{ki}h_i^{(l)}
\right|^2
]

Normalize:

[
\tilde w_{ki}
=============

\frac{w_{ki}^{MTO}}
{\sum_i w_{ki}^{MTO}+\epsilon}
]

---

## 16. Slot Intervention

To measure the effect of MTO slot (k) on property (p):

[
\mathcal O_k \leftarrow 0
]

[
\Delta y_{p,k}
==============

d(y_p,y_p^{(-k)})
]

Distance functions:

```text
mu: vector L2 distance
alpha: Frobenius distance
IR/Raman/UV spectrum: L1, L2, or cosine distance
scalar quantity: absolute difference
```

Property-specific atom map:

[
w_{p,i}
=======

\sum_k
\Delta y_{p,k}
\tilde w_{ki}
]

This gives a response-specific molecular fragment map.

---

## 17. Seed Stability

MTOs have permutation, sign, and basis freedoms. Therefore, do not compare slot (k) directly across different seeds.

Compare subspaces.

For each molecule and seed:

1. select top-r MTOs by activity or intervention importance;
2. build matrix (M^{(seed)}\in \mathbb R^{N\times r});
3. QR orthogonalize:

[
Q^{(seed)}=\mathrm{QR}(M^{(seed)})
]

4. compute subspace similarity:

[
S_{sub}(a,b)
============

\frac{
\mathrm{Tr}(Q_a Q_a^T Q_b Q_b^T)
}{r}
]

Range:

```text
0 = different subspaces
1 = identical subspaces
```

Outputs:

```text
outputs/metrics/seed_stability/subspace_similarity.csv
outputs/figures/debug/seed_stability_heatmap.pdf
outputs/reports/seed_stability.md
```

---

## 18. Stage-to-Stage Stability

Stage transfer:

```text
Stage A checkpoint
  → initialize Stage B
  → initialize Stage C
```

Rules:

1. Stage A best checkpoints are golden checkpoints.
2. Stage B/C must copy Stage A/B checkpoints, never overwrite them.
3. Stage B/C should use smaller learning rate for DetaNet backbone and MTO module.
4. Readout heads may use normal learning rate.
5. Stage-transfer analysis must compare MTO maps and subspaces across stages.

Suggested config:

```yaml
stage_transfer:
  freeze_backbone_first_epochs: 3
  backbone_lr_scale: 0.1
  mto_lr_scale: 0.5
  readout_lr_scale: 1.0
```

Metrics:

```text
Stage A mu/alpha map vs Stage B mu/alpha map
Stage B IR/Raman map vs Stage C IR/Raman map
Stage A/B/C subspace similarity
Prediction retention after adding new tasks
```

---

## 19. Chemical Validation

Use RDKit to annotate functional groups and fragments.

Initial SMARTS groups:

```python
FUNCTIONAL_GROUPS = {
    "carbonyl": "[CX3]=[OX1]",
    "aldehyde": "[CX3H1](=O)[#6,#1]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxyl": "C(=O)[OX2H1,OX1-]",
    "hydroxyl": "[OX2H]",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "aromatic_ring": "a1aaaaa1",
    "nitrile": "C#N",
    "ether": "[OD2]([#6])[#6]",
    "fluoro": "[F]",
}
```

Functional-group enrichment:

[
E_G
===

\frac{
\frac{1}{|G|}\sum_{i\in G}w_i
}{
\frac{1}{N}\sum_{i=1}^{N}w_i
}
]

Negative controls:

1. random atom mask;
2. shuffled atom map;
3. attention pooling baseline;
4. no-sign MTO baseline.

Outputs:

```text
outputs/metrics/chemical_validation/functional_group_enrichment.csv
outputs/figures/debug/chemical_validation/
outputs/reports/chemical_validation.md
```

---

## 20. Frozen Probe / Reusability

Frozen probe tests whether MTOs learned from early physical-response tasks can be reused for later tasks.

Protocol:

1. train Stage A on (\mu,\alpha);
2. freeze DetaNet backbone and MTO module;
3. train only new readout heads for Stage B or Stage C tasks;
4. compare with from-scratch and full fine-tuning models.

Interpretation:

```text
If frozen MTOs support non-random prediction of new response tasks, then MTOs are reusable molecule-level response representations rather than task-specific readout artifacts.
```

Outputs:

```text
outputs/metrics/frozen_probe/
outputs/figures/debug/frozen_probe/
outputs/reports/frozen_probe.md
```

---

## 21. Baselines and Ablations

Tier-1 baselines are required.

Tier-2 baselines are optional unless Tier-1 evidence is insufficient.

### 21.1 Tier-1 Baselines

1. DetaNet original readout;
2. positive attention pooling;
3. MTO without signed coefficient;
4. fixed-K MTO.

### 21.2 Tier-2 Baselines

1. no activity gate;
2. no diversity regularization;
3. multi-token pooling;
4. direct spectrum regression baseline;
5. frozen readout-only baseline.

### 21.3 Comparison Dimensions

Do not compare only MAE/RMSE.

Compare:

1. prediction metrics;
2. seed subspace stability;
3. stage-to-stage stability;
4. functional-group enrichment;
5. slot intervention effect;
6. frozen reuse;
7. memory cost;
8. runtime;
9. figure interpretability.

---

## 22. Dataset Split Rules

Dataset split must be locked before multi-seed training.

Rules:

1. If DetaNet provides an official QM9S split, reuse it.
2. If no official split is available locally, generate one split file once.
3. All seeds must use the same train/valid/test split.
4. Random seed should affect initialization and data shuffling, not dataset membership.
5. Store split files and split hash.

Outputs:

```text
outputs/splits/qm9s_split.json
outputs/splits/qm9s_split_hash.txt
```

---

## 23. HPC Rules for bjhpc_xxy_1

All remote operations must obey the user's CLAUDE.md.

### 23.1 Connection

Use only:

```bash
ssh bjhpc_xxy_1
```

Do not use:

```text
MCP
ssh_exec
ssh-mcp
MCP-based SSH workflows
deprecated aliases such as bjhpc or ustcA100
```

---

### 23.2 Workspace

All project operations must stay inside:

```bash
/data/home/scwc008/run/xxy
```

Before any remote project operation:

```bash
ssh bjhpc_xxy_1 'mkdir -p /data/home/scwc008/run/xxy && cd /data/home/scwc008/run/xxy && pwd'
```

Project directory:

```bash
/data/home/scwc008/run/xxy/MTO-Net
```

Forbidden remote roots include:

```text
/data/home/scwc008
/data/home/scwc008/run
/home/scwc008
/tmp
/etc
/usr
/opt
/var
```

Read-only system inspection is allowed for:

```text
hostname
whoami
module avail
nvidia-smi
sinfo
squeue
parajobs
scontrol show partition
scontrol show node
```

---

### 23.3 Safety for File Operations

Before destructive or broad commands, print:

```bash
pwd
readlink -f <target>
```

Then verify target begins with:

```bash
/data/home/scwc008/run/xxy
```

Forbidden examples:

```bash
rm -rf /data/home/scwc008/*
rm -rf /data/home/scwc008/run/*
find /data/home/scwc008 -name ...
chmod -R ... /data/home/scwc008
mv ~/... .
cp /etc/... .
```

Dataset directories are read-only unless the user explicitly says otherwise.

---

### 23.4 GPU Priority

GPU priority:

```text
H200 first
H100 second
A800 final fallback
```

Do not hard-code Slurm GPU flags before detection.

Phase 0 must inspect:

```bash
sinfo -o "%P %G %D %t"
scontrol show partition
scontrol show node | grep -i -E "h200|h100|a800|gres|partition"
```

Then generate actual Slurm templates.

Do not assume:

```bash
#SBATCH -p gpu
#SBATCH --gpus=1
```

selects H200.

Use the detected partition/GRES format.

---

### 23.5 Slurm

Long training must not run on login node.

GPU training must use Slurm.

Allowed job types:

1. smoke test job;
2. small/medium sanity job;
3. Stage A multi-seed job array;
4. Stage B multi-seed job array;
5. Stage C multi-seed job array;
6. baseline/ablation jobs;
7. analysis jobs if GPU is needed.

Prefer one seed per GPU unless DDP is already stable.

---

### 23.6 Environment

Do not assume the TranSpec environment is valid for MTO-Net.

Phase 0 may inspect existing environments. Before training, use or create a dedicated environment such as:

```text
mto
detanet
```

If only an existing environment is available, all required imports must pass before training.

Required environment audit:

```bash
which python
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import numpy, scipy, pandas, matplotlib"
python -c "import rdkit"
python -c "import torch_geometric"
```

Additional imports depend on the actual DetaNet requirements.

Save:

```text
outputs/reports/00_env.md
outputs/reports/pip_freeze.txt
outputs/reports/conda_env_export.yml
```

---

## 24. Execution Phases

## Phase 0: Audit Only

Goal:

```text
Inspect environment, repo, data, DetaNet forward path, labels, and GPU resources.
```

Do not write MTO model code yet.

Tasks:

1. connect with `ssh bjhpc_xxy_1`;
2. verify workspace;
3. check hostname/whoami/pwd;
4. inspect GPU and Slurm;
5. detect H200/H100/A800 request format;
6. inspect MTO-Net repo;
7. confirm TOTAL.md and PROJECT.md exist;
8. inspect DetaNet source code;
9. inspect local QM9S dataset path;
10. list available dataset labels;
11. run minimal DetaNet import;
12. run minimal DetaNet forward if possible;
13. identify atom-level representation hook point;
14. inspect Python environment;
15. write audit reports.

Outputs:

```text
outputs/reports/00_env.md
outputs/reports/00_repo_audit.md
outputs/reports/00_dataset_audit.md
outputs/reports/00_detanet_forward.md
outputs/reports/00_slurm_gpu.md
scripts/slurm/template_detected_gpu.sh
```

Acceptance criteria:

1. workspace confirmed;
2. repo found or cloned inside workspace;
3. dataset located;
4. label fields listed;
5. DetaNet import tested;
6. atom representation hook point identified;
7. GPU request format documented;
8. no files outside workspace modified.

---

## Phase 1: Complete Codebase Implementation

Goal:

```text
Implement the complete MTO-Net code path before smoke tests.
```

Tasks:

1. implement DetaNet adapter;
2. implement valence electron counter;
3. implement (K=N_{\mathrm{val}}) MTO bank;
4. implement signed MTO assembly;
5. implement activity gate;
6. implement response readouts;
7. implement physical quantity heads based on available labels;
8. implement spectrum builder;
9. implement Stage A/B/C training scripts;
10. implement evaluation scripts;
11. implement MTO cache;
12. implement seed stability analysis;
13. implement stage-transfer analysis;
14. implement chemical validation;
15. implement slot intervention;
16. implement frozen probe;
17. implement baselines and Tier-1 ablations;
18. implement plotting scripts;
19. implement Slurm scripts;
20. implement tests.

Outputs:

```text
mtonet/
scripts/
configs/
tests/
outputs/reports/01_implementation_summary.md
```

Acceptance criteria:

1. all expected files created;
2. code imports successfully;
3. configs exist;
4. scripts have `--help` or documented CLI;
5. no long training submitted during implementation;
6. git diff summarized.

---

## Phase 2: Full Smoke Test

Goal:

```text
Verify the full project path on a tiny subset.
```

Suggested setting:

```text
molecules: 32
epochs: 2
seeds: 0 and 1
GPU: 1
```

Smoke test must verify:

1. data loading;
2. DetaNet forward;
3. atom representation hook;
4. MTO bank construction;
5. prediction heads;
6. physical response outputs;
7. loss computation;
8. backward pass;
9. optimizer step;
10. checkpoint save;
11. checkpoint reload;
12. MTO cache save;
13. evaluation;
14. seed subspace similarity on toy data;
15. one molecule visualization;
16. report generation.

Outputs:

```text
outputs/smoke/
outputs/reports/02_smoke_test.md
outputs/figures/debug/smoke/
```

Acceptance criteria:

1. no NaN;
2. loss decreases or remains finite;
3. checkpoint exists;
4. reload works;
5. MTO cache exists;
6. at least one MTO heatmap generated;
7. stability script runs;
8. no workspace violation.

---

## Phase 3: Small/Medium Sanity Run

Goal:

```text
Verify training stability before full QM9S.
```

Suggested setting:

```text
train: 5k-20k molecules
valid: 1k
test: 1k
seeds: 0 and 1
```

Tasks:

1. run Stage A medium;
2. optionally run Stage B medium if labels are available;
3. monitor loss;
4. inspect MTO activity distribution;
5. inspect MTO collapse;
6. inspect cache size;
7. inspect memory usage;
8. generate debug figures.

Outputs:

```text
outputs/medium/
outputs/reports/03_medium_sanity.md
outputs/figures/debug/medium/
```

Acceptance criteria:

1. training stable;
2. memory acceptable;
3. no obvious MTO collapse;
4. cache size controlled;
5. seed similarity computable;
6. decision made to proceed to full QM9S.

---

## Phase 4: Full QM9S Stage A Multi-Seed

Goal:

```text
Train Stage A on full QM9S and establish MTO seed stability.
```

Tasks:

1. use locked split;
2. train Stage A with seeds 0,1,2,3,4;
3. use one H200 per seed if available;
4. evaluate all seeds;
5. save best checkpoints;
6. compute seed subspace similarity;
7. generate Stage A figures;
8. write report.

Outputs:

```text
outputs/checkpoints/stage_a_seed0/
outputs/checkpoints/stage_a_seed1/
outputs/checkpoints/stage_a_seed2/
outputs/checkpoints/stage_a_seed3/
outputs/checkpoints/stage_a_seed4/
outputs/metrics/stage_a/
outputs/metrics/seed_stability/
outputs/reports/04_stage_a_full.md
```

Acceptance criteria:

1. at least 4 of 5 seeds finish successfully;
2. (\mu,\alpha) metrics are reasonable;
3. MTO cache generated;
4. seed subspace similarity computed;
5. representative maps generated;
6. Stage A checkpoints protected as golden checkpoints.

---

## Phase 5: Full QM9S Stage B

Goal:

```text
Add vibrational response tasks and test MTO reuse.
```

Tasks:

1. initialize from Stage A checkpoints;
2. train Stage B with seeds 0,1,2,3,4;
3. enable only available physical labels;
4. generate IR/Raman from physical quantities whenever possible;
5. evaluate prediction metrics;
6. compute Stage A vs Stage B stability;
7. analyze whether (\mu,\alpha) maps remain stable;
8. write report.

Outputs:

```text
outputs/checkpoints/stage_b_seed*/
outputs/metrics/stage_b/
outputs/metrics/stage_transfer_ab/
outputs/reports/05_stage_b_full.md
```

Acceptance criteria:

1. Stage B training stable;
2. Stage A checkpoints not overwritten;
3. physical-response outputs generated when labels exist;
4. Stage A/B stability computed;
5. no catastrophic collapse of MTO maps.

---

## Phase 6: Full QM9S Stage C

Goal:

```text
Add UV/electronic-response task and test broader reuse.
```

Tasks:

1. initialize from Stage B checkpoints;
2. train Stage C with seeds 0,1,2,3,4;
3. enable available UV-related labels;
4. use broadened UV spectrum only if transition-level labels are unavailable or unstable;
5. evaluate all tasks;
6. compute Stage A/B/C stability;
7. analyze electronic-response maps;
8. write report.

Outputs:

```text
outputs/checkpoints/stage_c_seed*/
outputs/metrics/stage_c/
outputs/metrics/stage_transfer_abc/
outputs/reports/06_stage_c_full.md
```

Acceptance criteria:

1. Stage C training stable;
2. UV/electronic-response supervision documented;
3. Stage A/B/C stability computed;
4. representative maps generated;
5. conclusions remain cautious.

---

## Phase 7: Frozen Probe

Goal:

```text
Test whether MTOs are reusable across tasks.
```

Tasks:

1. load Stage A checkpoints;
2. freeze DetaNet backbone and MTO module;
3. train only new readout heads for Stage B/C tasks;
4. compare frozen probe with full fine-tuning and from-scratch training;
5. write report.

Outputs:

```text
outputs/checkpoints/frozen_probe/
outputs/metrics/frozen_probe/
outputs/reports/07_frozen_probe.md
```

Acceptance criteria:

1. frozen parameters verified;
2. readout-only training works;
3. new task performance above random/non-informative baseline;
4. reusability conclusion supported.

---

## Phase 8: Baselines and Ablations

Goal:

```text
Show MTO is not ordinary pooling or attention artifact.
```

Required Tier-1 runs:

1. DetaNet original readout;
2. attention pooling;
3. MTO without signed coefficient;
4. fixed-K MTO.

Optional Tier-2 runs:

1. no activity gate;
2. no diversity loss;
3. direct spectrum regression;
4. multi-token pooling.

Outputs:

```text
outputs/checkpoints/ablations/
outputs/metrics/ablations/
outputs/reports/08_baselines_ablations.md
```

Acceptance criteria:

1. Tier-1 baselines completed;
2. comparison includes prediction and interpretability metrics;
3. Full MTO has stronger stability/interpretability evidence than pooling baselines.

---

## Phase 9: Final Analysis and Figures

Goal:

```text
Generate final paper-ready figures and reports.
```

Only after metrics are frozen, call:

```text
/nature-skills:nature-figure
```

Do not generate final Nature-style figures before training results are stable.

Figure plan:

```text
Figure 1: Concept and architecture
Figure 2: DetaNet + MTO module
Figure 3: Stage A prediction and seed stability
Figure 4: Chemical validation and functional-group maps
Figure 5: Stage-to-stage stability
Figure 6: Frozen probe and reusability
Figure 7: Baselines and ablations
Extended Data: additional examples, failure cases, environment, data statistics
```

Outputs:

```text
outputs/figures/final/
outputs/figures/source_data/
outputs/reports/09_final_figures.md
outputs/reports/FINAL_PROJECT_REPORT.md
```

Each final figure must have source data.

---

## 25. Slurm Script Generation

Do not hard-code H200 parameters before detection.

Claude Code must generate Slurm scripts after Phase 0 GPU audit.

Template logic:

```text
if H200 partition/GRES available:
    use H200
elif H100 available:
    use H100
else:
    use A800
```

Example generic script skeleton:

```bash
#!/bin/bash
#SBATCH -J mto_job
#SBATCH --time=48:00:00
#SBATCH --output=outputs/logs/slurm/%x_%j.out
#SBATCH --error=outputs/logs/slurm/%x_%j.err

set -euo pipefail

cd /data/home/scwc008/run/xxy/MTO-Net

module load miniforge3/24.11

if conda env list | grep -q "mto"; then
  source activate mto
elif conda env list | grep -q "detanet"; then
  source activate detanet
else
  echo "No valid MTO/DetaNet conda environment found."
  hostname
  whoami
  pwd
  module avail || true
  conda env list || true
  exit 1
fi

hostname
whoami
pwd
which python
python --version
nvidia-smi || true

python <SCRIPT> <ARGS>
```

The actual GPU lines must be inserted by the detector script.

---

## 26. Metrics Format

All metrics must be saved as CSV and JSON when possible.

### 26.1 Prediction Metrics

Path:

```text
outputs/metrics/<stage>/prediction_metrics.csv
outputs/metrics/<stage>/prediction_metrics.json
```

Fields:

```text
stage
seed
task
mae
rmse
r2
spearman
num_train
num_valid
num_test
checkpoint
git_commit
split_hash
```

---

### 26.2 Seed Stability Metrics

Path:

```text
outputs/metrics/seed_stability/subspace_similarity.csv
```

Fields:

```text
stage
mol_id
seed_a
seed_b
top_r
similarity
property
selection_method
```

---

### 26.3 Chemical Validation Metrics

Path:

```text
outputs/metrics/chemical_validation/functional_group_enrichment.csv
```

Fields:

```text
stage
seed
mol_id
smiles
property
functional_group
num_group_atoms
enrichment
random_enrichment_mean
random_enrichment_std
p_value
```

---

### 26.4 Intervention Metrics

Path:

```text
outputs/metrics/intervention/slot_intervention.csv
```

Fields:

```text
stage
seed
mol_id
property
slot_k
activity
delta_y
normalized_delta_y
top_atoms
```

---

## 27. Report Format

Every phase must generate a report.

Report template:

```markdown
# Report Title

## Environment
- hostname
- whoami
- pwd
- python
- cuda
- gpu
- git commit

## Commands Executed

## Files Created or Modified

## Jobs Submitted

## Results

## Metrics

## Figures

## Problems Encountered

## Next Actions

## Safety Confirmation
No files outside /data/home/scwc008/run/xxy were modified.
```

Final report:

```text
outputs/reports/FINAL_PROJECT_REPORT.md
```

The final report must include:

1. project summary;
2. implementation summary;
3. environment;
4. dataset fields;
5. Stage A results;
6. seed stability;
7. chemical validation;
8. Stage B/C transfer;
9. frozen probe;
10. baselines/ablations;
11. final figures;
12. failure cases;
13. next steps;
14. all important paths.

---

## 28. Minimum Success Criteria

The project is minimally successful if:

1. DetaNet atom-level representation is successfully hooked;
2. MTO bank with (K=N_{\mathrm{val}}) is implemented;
3. signed MTO assembly works;
4. Stage A trains on full QM9S;
5. Stage A multi-seed stability can be measured;
6. MTO maps are stable at the subspace level across seeds;
7. MTO maps show chemically meaningful enrichment for at least several functional groups/fragments;
8. Stage B/C can reuse Stage A MTOs without complete collapse;
9. frozen probe gives non-random transfer to new response tasks;
10. Full MTO beats simple pooling baselines on stability or interpretability metrics;
11. all code, logs, checkpoints, metrics, cache, and figures are reproducible;
12. all HPC operations obey workspace restrictions.

---

## 29. Risk Management

### Risk 1: DetaNet Hidden Representation Is Hard to Hook

Response:

1. inspect source carefully;
2. add `return_atom_features=True`;
3. preserve original forward behavior;
4. use adapter;
5. write forward tests.

---

### Risk 2: QM9S Labels Are Not What We Expect

Response:

1. audit local dataset first;
2. enable tasks only when labels exist;
3. do not fabricate labels;
4. document skipped targets;
5. use available physical quantities first.

---

### Risk 3: MTO Collapse

Symptoms:

1. all slots attend to same atoms;
2. activity all near 0 or all near 1;
3. no diversity across slots.

Response:

1. inspect activity distribution;
2. add or tune diversity loss;
3. tune signed coefficient;
4. tune slot embedding;
5. compare no-sign and fixed-K ablations.

---

### Risk 4: K=Nval Causes Memory Pressure

Response:

1. inspect K distribution;
2. reduce batch size;
3. use gradient accumulation;
4. use mixed precision if safe;
5. use `k_max_cap` only for debugging, not final main experiments unless necessary.

---

### Risk 5: Stage B/C Destroy Stage A MTOs

Response:

1. use lower learning rate for backbone;
2. freeze backbone for early epochs;
3. compare stage stability;
4. use frozen probe;
5. do not overwrite Stage A checkpoints.

---

### Risk 6: Chemical Validation Is Weak

Response:

1. focus on clearer functional groups first;
2. use property-specific intervention maps;
3. compare against negative controls;
4. analyze failure cases honestly;
5. do not overclaim physical orbital identity.

---

### Risk 7: H200 Unavailable

Response:

1. try H100;
2. then A800;
3. use job arrays;
4. use one seed per GPU;
5. do not run long training on login node.

---

## 30. First Actual Task for Claude Code

Claude Code must begin with:

```text
Phase 0 audit only.
```

Do not write MTO code until Phase 0 is complete.

Phase 0 checklist:

1. connect to `bjhpc_xxy_1`;
2. enter `/data/home/scwc008/run/xxy`;
3. verify `pwd`;
4. inspect GPU and Slurm;
5. inspect or clone MTO-Net repo inside workspace;
6. confirm `TOTAL.md` and `PROJECT.md`;
7. inspect DetaNet code;
8. inspect QM9S data path;
9. list QM9S label fields;
10. test Python environment;
11. test DetaNet import;
12. identify atom-level representation hook point;
13. generate Slurm GPU template;
14. write all audit reports;
15. stop and report to user.

Do not submit long training jobs in Phase 0.

---

## 31. Final One-Sentence Execution Rule

```text
First audit the environment and data; then implement the complete MTO-Net codebase; then run full smoke tests; then run small/medium sanity checks; then train Stage A/B/C on full QM9S with multiple seeds; finally perform stability, chemical, physical-response, and reusability analyses with paper-ready figures.
```
