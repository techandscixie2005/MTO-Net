# MTO-Net Design (Minimal Viable Loop)

## Architecture

```
Input (z, pos)
    |
    v
DetaNet Backbone (unchanged)
  - 3 Interaction_Block layers
  - maxl=3 (scalar + l=1,2,3 irrep tensors)
  - out_type='latent' → returns (S, T)
    |
    v
DetaNet Adapter
  - Thin wrapper: exposes S as atom_features
  - S: per-atom scalar features [N_atoms, num_features]
  - T: per-atom irrep tensor [N_atoms, vdim] (unused for now)
    |
    v
ValenceAdaptiveMTO
  - K = sum of valence electrons per molecule
  - Slot embedding (reuses atom type embedding)
  - Route MLP: softmax over atoms → a_{k,i}
  - Sign MLP: tanh → s_{k,i}
  - c_{k,i} = normalize(|a_{k,i} * s_{k,i}|)
  - O_k = sum_i c_{k,i} * atom_feature_i
    |
    v
Output: {O, coeff, mask, atom_mask, K_per_mol}
```

## Hook Location

- **File**: `third_party/DetaNet/detanet_model/detanet.py:337-341`
- **Method**: `DetaNet.forward()` with `out_type='latent'`
- **Hook point**: After `Interaction_Block` loop, `(S, T)` is returned directly
  without going through output layers (`sout`, `tout`)
- **S**: scalar atom features [N, 128] — used as MTO input
- **T**: irrep tensor features [N, vdim] — available for l=1/l=2 relaxation

## DetaNet Backbone (unchanged)

DetaNet is used as-is. The adapter calls `DetaNet(z, pos, batch)` with
`out_type='latent', summation=False, scalar_outsize=0`.

## MTO Formula

For molecule with K valence electrons and N atoms:

```
a_{k,i} = softmax_i(MLP_route([f_i, type_emb(z_i), slot_emb(k)]))
s_{k,i} = tanh(MLP_sign([f_i, type_emb(z_i), slot_emb(k)]))
c_{k,i} = a_{k,i} * s_{k,i} / sum_i |a_{k,i} * s_{k,i}|
O_k = sum_i c_{k,i} * f_i
```

where:
- f_i: atom feature vector [C]
- k in [0, K-1]: slot index
- i in [0, N-1]: atom index

## Current Limitations

1. **Scalar-only**: T tensor (irreps l=1,2,3) is available but unused
2. **No pretrained weights**: DetaNet backbone randomly initialized
3. **Random features fallback**: Smoke test defaults to random atom features
4. **Synthetic molecules only**: QM9S download pending
5. **No charge correction**: Neutral molecules only
6. **K may have extra slots**: K = total valence count, not occupied orbitals
7. **MTO is NOT a physical molecular orbital**: It is an orbital-like latent
   mode in the DetaNet equivariant hidden space
