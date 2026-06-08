#!/usr/bin/env python3
"""Smoke test for MTO forward pass using synthetic molecules."""

import json
import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.mto.molecule_builder import make_ethanol, make_formaldehyde, make_water
from src.mto.mto_module import ValenceAdaptiveMTO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-mols", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--use-detanet", action="store_true", default=False)
    args = parser.parse_args()

    print("=== MTO Smoke Forward Test ===")

    mol_data = [make_ethanol(), make_formaldehyde(), make_water()]

    all_z = torch.cat([m[0] for m in mol_data])
    all_pos = torch.cat([m[1] for m in mol_data])
    batches = []
    for bi, m in enumerate(mol_data):
        batches.append(torch.full((len(m[0]),), bi, dtype=torch.long))
    all_batch = torch.cat(batches)
    labels = ["ethanol", "formaldehyde", "water"]

    print(f"Batch: {len(mol_data)} molecules, {len(all_z)} atoms")

    C = args.feature_dim

    if args.use_detanet:
        print("\nLoading DetaNet backbone...")
        from src.mto.compat import *
        from src.mto.detanet_adapter import DetaNetBackboneAdapter
        device = torch.device("cpu")
        backbone = DetaNetBackboneAdapter(
            num_features=C, maxl=1, num_block=1, rc=5.0, device=device
        )
        backbone.eval()
        with torch.no_grad():
            result = backbone(z=all_z, pos=all_pos, batch=all_batch)
        atom_features = result["atom_features"]
        print(f"DetaNet atom_features shape: {list(atom_features.shape)}")
    else:
        print("\nUsing random synthetic atom features (no DetaNet backbone)")
        atom_features = torch.randn(len(all_z), C)

    print(f"atom_features shape: {list(atom_features.shape)}")

    mto = ValenceAdaptiveMTO(feature_dim=C)
    out = mto(atom_features=atom_features, z=all_z, batch=all_batch)

    K_per_mol = out["K_per_mol"]
    O = out["O"]
    coeff = out["coeff"]
    mask = out["mask"]
    atom_mask = out["atom_mask"]

    print(f"\nK_per_mol: {K_per_mol.tolist()}")
    for bi in range(len(mol_data)):
        Kb = int(K_per_mol[bi])
        n_atoms = int(atom_mask[bi].sum().item())
        print(f"  mol[{bi}] {labels[bi]}: K={Kb}, atoms={n_atoms}")

    print(f"\nO shape: {list(O.shape)}")
    print(f"coeff shape: {list(coeff.shape)}")
    print(f"mask shape: {list(mask.shape)}")

    has_nan = False
    for key in ["O", "coeff"]:
        if torch.isnan(out[key]).any():
            print(f"  NaN detected in {key}!")
            has_nan = True
    if not has_nan:
        print("any NaN? False")

    summary = {
        "num_mols": len(mol_data),
        "num_atoms": int(len(all_z)),
        "feature_dim": C,
        "atom_features_shape": list(atom_features.shape),
        "K_per_mol": K_per_mol.tolist(),
        "O_shape": list(O.shape),
        "coeff_shape": list(coeff.shape),
        "any_nan": has_nan,
        "molecules": labels,
        "synthetic": True,
    }
    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/smoke_mto_forward.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: outputs/metrics/smoke_mto_forward.json")

    print("\n=== Smoke test PASSED ===")
    return out, labels


if __name__ == "__main__":
    main()
