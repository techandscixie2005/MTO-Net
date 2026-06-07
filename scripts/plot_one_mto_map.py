#!/usr/bin/env python3
"""Generate one MTO atom-contribution map figure."""

import sys
import os
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.mto.mto_module import ValenceAdaptiveMTO
from src.mto.visualization import plot_mto_map


def make_ethanol():
    z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1])
    pos = torch.tensor([
        [-0.750, 0.000, 0.000],
        [0.750, 0.000, 0.000],
        [1.450, 1.390, 0.000],
        [-1.200, 0.940, 0.000],
        [-1.200, -0.500, -0.880],
        [-1.200, -0.500, 0.880],
        [1.200, -0.500, -0.880],
        [1.200, -0.500, 0.880],
        [2.440, 1.240, 0.000],
    ], dtype=torch.float32)
    batch = torch.zeros(9, dtype=torch.long)
    return z, pos, batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol-index", type=int, default=0)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--feature-dim", type=int, default=128)
    args = parser.parse_args()

    print("=== Generate One MTO Map ===\n")
    print(f"Molecule index: {args.mol_index}, Slot: {args.slot}")

    torch.manual_seed(42)
    z, pos, batch = make_ethanol()
    label = "ethanol"
    is_synthetic = True
    C = args.feature_dim

    atom_features = torch.randn(len(z), C)

    mto = ValenceAdaptiveMTO(feature_dim=C)
    mto.eval()
    with torch.no_grad():
        out = mto(atom_features=atom_features, z=z, batch=batch)

    K_per_mol = out["K_per_mol"]
    K = int(K_per_mol[0])
    coeff = out["coeff"][0, args.slot, :int(z.size(0))]

    print(f"Molecule: {label}, K={K}")
    coeff_list = coeff.tolist()
    print(f"Coefficients for slot {args.slot}: {[round(v, 4) for v in coeff_list]}")
    print(f"Sum|c|: {coeff.abs().sum().item():.4f}")

    os.makedirs("outputs/figures", exist_ok=True)
    save_path = "outputs/figures/one_mto_map.png"
    plot_mto_map(
        pos=pos,
        z=z,
        coeff=coeff,
        mol_label=label,
        slot_idx=args.slot,
        K=K,
        is_synthetic=is_synthetic,
        save_path=save_path,
    )

    print(f"\n=== Map generated: {save_path} ===")
    return save_path


if __name__ == "__main__":
    main()
