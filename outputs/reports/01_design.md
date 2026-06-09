# MTO-Net Design Document

## 1. Overall Architecture

```
Input {Z_i, r_i}
  -> DetaNet atom-level equivariant tensor encoder
  -> Valence-adaptive MTO bank (K = N_val)
  -> Signed center-free MTO assembly
  -> Activity gate
  -> Multi-head response readouts
  -> Predictions {mu, alpha, IR, Raman, UV}
```

## 2. DetaNet Adapter

**File:** `src/mto/detanet_adapter.py`

The `DetaNetBackboneAdapter` wraps `DetaNet(out_type="latent", summation=False, scale=None)`.

Hook point: After `Interaction_Block` layers, DetaNet returns `(S, T)` where:
- `S`: [N_atoms, 128] invariant scalar features
- `T`: [N_atoms, 1920] equivariant irrep tensor (maxl=3, vdim=1920)

`S` (scalar features) serves as primary MTO input. `T` is stored for reference.

Key design decisions:
- `device=None` in adapter defaults to CPU; Trainer moves to GPU
- Compat patches fix pyg-lib/e3nn issues with PyTorch 2.8

## 3. Valence-Adaptive MTO Bank

**File:** `src/mto/valence.py`

```
VALENCE_ELECTRONS = {1:1, 6:4, 7:5, 8:6, 9:7}  # H, C, N, O, F
K_b = sum_i valence(z_i) for molecule b
K_max = max_b K_b
mto_mask[b, k] = (k < K_b)
```

MVP: neutral closed-shell molecules only. No charge correction.

## 4. Signed Center-Free MTO Assembly

**File:** `src/mto/mto_module.py`

```
e_ki = MLP_r(h0_i, slot_emb(k))          # routing score
a_ki = softmax_i(e_ki)                     # attention
s_ki = tanh(MLP_s(h0_i, slot_emb(k)))     # signed coefficient
c_ki = normalize(a_ki * s_ki)             # normalized L1 sum per slot
O_k = sum_i c_ki * h_i                    # MTO construction
```

Properties:
- `c_ki` is scalar/invariant — preserves equivariance
- L1 normalization: sum_i|c_ki| = 1 per slot
- Center-free: no geometric center coordinates
- Signed: allows bonding/antibonding-like patterns

## 5. Activity Gate

**File:** `src/mto/activity_gate.py`

Three modes:
- `none`: O_tilde_k = O_k
- `simple`: g_k = sigmoid(MLP_g(O_k)), O_tilde_k = g_k * O_k
- `fermi_dirac`: n_k = 2*sigmoid((mu-eps_k)/theta), charge-conserving

Default: `simple`. `fermi_dirac` requires implicit differentiation.

## 6. Response Readouts

**File:** `src/mto/mto_readout.py`

| Task | Architecture | Output |
|------|-------------|--------|
| mu | slot_pool + MLP(128->64->3) | [B, 3] |
| alpha | slot_pool + MLP(128->64->9) | [B, 9] (upper triangular) |
| ir | slot_pool + MLP(128->128->3501) | [B, 3501] |
| raman | slot_pool + MLP(128->128->3501) | [B, 3501] |
| uv | slot_pool + MLP(128->128->701) | [B, 701] |

Tasks auto-enabled based on label availability (`auto` in config).

## 7. Losses

**File:** `src/mto/losses.py`

- mu: MSE on standardized dipole vector
- alpha: MSE on flattened 3x3 polarizability
- IR/Raman/UV: MSE + cosine spectral loss (weight 0.1)
- Diversity loss: λ_div * sum_{k!=k'} cos_sim(O_k, O_k')^2
- Entropy loss: λ_ent * routing entropy penalty

Default weights: mu=1.0, alpha=1.0, ir=0.3, raman=0.3, uv=0.1

## 8. Training Stages

### Stage A (mu, alpha)
- Foundation: fundamental response properties
- 50 epochs, lr=1e-3, batch_size=8 or 64
- 5 seeds for stability analysis

### Stage B (+IR, Raman)
- Initialize from Stage A best checkpoint
- Adds vibrational spectral prediction
- 100 epochs, lr=5e-4

### Stage C (+UV)
- Initialize from Stage B best checkpoint
- Full response property prediction
- 100 epochs, lr=3e-4

## 9. Analysis Pipeline

### Seed subspace stability
MTO slots have sign/permutation freedom -> compare subspaces via QR projection similarity.

### Slot intervention
Zero one MTO slot -> measure Δ prediction per property.

### Functional group analysis
RDKit SMARTS patterns -> enrichment of atom contributions in functional groups.

### Frozen probe
Freeze backbone + MTO module, train only new readout heads.

## 10. Baselines

| Baseline | Description |
|----------|------------|
| Direct readout | DetaNet -> sum pool -> MLP |
| Attention pooling | DetaNet -> attention -> pool -> MLP |
| MTO without sign | Softmax-only (no tanh sign) |
| Fixed-K MTO | K=20 fixed slots |
| Full MTO | Signed, valence-adaptive |

## 11. Limitations

- Only H, C, N, O, F supported (valence table)
- No charge correction (neutral molecules only)
- MTO routing uses only scalar features (not full tensor)
- DetaNet representation is flat (not tensor-order split)
- Spectra bin sizes hardcoded (3501 IR/Raman, 701 UV)
- K=N_val can be large for big molecules
- No E/F/Hessian/MD in current version
