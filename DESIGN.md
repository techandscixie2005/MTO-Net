# MTO-Net Design Document

## DetaNet Backbone Adapter

**File:** `src/mto/detanet_adapter.py`

DetaNet is imported from `third_party/DetaNet/` via `sys.path` insertion. The adapter wraps
`DetaNet(out_type="latent", summation=False, scale=None)` and returns:

```python
{
    "atom_tensors": {"h0": S, "T": T},
    "atom_features": S,   # [N_atoms, num_features] — primary MTO input
    "batch": batch,
    "z": z,
    "pos": pos,
}
```

**Hook point:** After the `Interaction_Block` loop in DetaNet.forward, when `out_type=latent`,
the model returns `(S, T)` — per-atom scalar features and irrep tensors.

**Key compatibility patches:**
- `src/mto/compat.py`: Patches `torch_geometric.nn.radius_graph` to use `torch_cluster.radius_graph`
  (pyg-lib >= 0.6 not available for torch 2.7+cu128)
- `torch.serialization.add_safe_globals([slice])`: Fixes e3nn 0.4.4 with torch 2.7 `weights_only=True`
- `scale=None` in DetaNet init: Avoids `tuple * float` bug when `out_type=latent`

## Hooked Tensor Names and Shapes

| Name | Shape | Description |
|------|-------|-------------|
| S (h₀) | [N, 128] | Invariant scalar atom features |
| T | [N, 1920] | Equivariant irrep tensor (maxl=3, vdim=1920) |
| O | [B, K_max, 128] | MTO slot features |

MTO routing uses **only scalar/invariant** features (S/h₀). T is passed through for reference.

## Valence-Adaptive K

**File:** `src/mto/valence.py`

```python
VALENCE_ELECTRONS = {1:1, 6:4, 7:5, 8:6, 9:7}  # H, C, N, O, F

K_b = sum_i valence(z_i) for molecule b
K_max = max(K_b)
mto_mask[b, k] = (k < K_b)
```

No charge correction in v1. Only H, C, N, O, F are supported.

## Signed Atom-to-MTO Assembly

**File:** `src/mto/mto_module.py`

```
a_ki = softmax_i(route_mlp([h₀_i, atom_emb(z_i), slot_emb(k)]))
s_ki_l = tanh(sign_mlp_l([h₀_i, atom_emb(z_i), slot_emb(k)]))
c_ki_l = normalize(a_ki * s_ki_l)     # sum_i |c_ki| = 1 per slot
O_k_l = sum_i c_ki_l * h_i_l
```

Key properties:
- `c_ki_l` is scalar/invariant
- Variable K per molecule (based on valence electrons)
- Variable atom counts within batch
- Normalization: `sum_i |c_ki| = 1` per slot

## Readout Heads

**File:** `src/mto/mto_readout.py`

| Stage | Tasks | Output dims |
|-------|-------|------------|
| A | mu, alpha | 3, 6 |
| B | mu, alpha, IR, Raman | 3, 6, 3501, 3501 |
| C | mu, alpha, IR, Raman, UV | 3, 6, 3501, 3501, 601 |

Each head is a 2-layer MLP: `Linear(feature_dim, hidden) → SiLU → Linear(hidden, out_dim)`.
Pooling across slots: `sum_k head(O_k)`.

## Losses

**File:** `src/mto/losses.py`

| Property | Loss |
|----------|------|
| mu | MSE on standardized dipole vector |
| alpha | Frobenius MSE on 3×3 polarizability matrix |
| IR/Raman/UV | MSE + cosine spectral loss (weight=0.1) |

Default task weights: mu=1.0, alpha=1.0, ir=0.3, raman=0.3, uv=0.1

Labels are standardized on the training split (mean=0, std=1) and restored for evaluation.

## Training Stages

### Stage A (mu, alpha)
- 50 epochs, lr=1e-3, batch_size=8
- Baseline: dipole + polarizability prediction
- Smoke: 32 mols, Medium: 512, Full: 130k

### Stage B (+IR, Raman)
- 100 epochs, lr=5e-4, batch_size=8
- Initialize from Stage A best checkpoint
- Adds vibrational spectral prediction

### Stage C (+UV)
- 100 epochs, lr=3e-4, batch_size=8
- Initialize from Stage B best checkpoint
- Full response property prediction

## Stability Metrics

### Seed Subspace Stability
MTO slots have permutation/sign/gauge freedom — do NOT compare slot-0 to slot-0 directly.
Instead: `S_sub = trace(Q_a Q_a^T Q_b Q_b^T) / r` where Q_a, Q_b are QR-orthogonalized bases.

### Stage Stability
Pearson correlation of atom contribution maps: `corr(w_mu_stage_A, w_mu_stage_B)`

### Slot Intervention
Zero one MTO slot at a time, measure `Δmu, Δalpha, ΔIR, ΔRaman, ΔUV`.

## Baselines

| Baseline | Description |
|----------|------------|
| Direct readout | DetaNet → sum pool → MLP |
| Sum pooling | Same as direct readout |
| Attention pooling | DetaNet → learned attention → pool → MLP |
| Fixed-K token | K=20 fixed slots (no valence adaptivity) |
| MTO without sign | Softmax-only routing (no `tanh(sign_mlp)`) |
| Full MTO | Signed, valence-adaptive MTO |

## Limitations

- Only H, C, N, O, F elements supported (valence table limited)
- No charge correction
- MTO routing uses only invariant scalars (not full equivariant tensors)
- K_max can be large for big molecules (performance linear in max valence)
- Spectra bin sizes are hardcoded (3501 IR/Raman, 601 UV)
- No E/F/Hessian/MD prediction in current version
