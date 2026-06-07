#!/usr/bin/env python3
"""Generate one MTO atom-contribution map figure."""

import os
import argparse
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.mto.molecule_builder import make_ethanol
from src.mto.mto_module import ValenceAdaptiveMTO
from src.mto.visualization import plot_mto_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol-index", type=int, default=0)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--feature-dim", type=int, default=128)
    args = parser.parse_args()

    print("=== Generate One MTO Map ===")
    print(f"Molecule index: {args.mol_index}, Slot: {args.slot}")

    torch.manual_seed(42)
    z, pos, batch = make_ethanol()
    label = "ethanol"
    C = args.feature_dim

    atom_features = torch.randn(len(z), C)

    mto = ValenceAdaptiveMTO(feature_dim=C)
    mto.eval()
    with torch.no_grad():
        out = mto(atom_features=atom_features, z=z, batch=batch)

    K = int(out["K_per_mol"][0])
    coeff = out["coeff"][0, args.slot, :int(z.size(0))]

    print(f"Molecule: {label}, K={K}")
    coeff_list = coeff.tolist()
    print(f"Coefficients for slot {args.slot}: {[round(v, 4) for v in coeff_list]}")
    print(f"Sum|c|: {coeff.abs().sum().item():.4f}")

    os.makedirs("outputs/figures", exist_ok=True)
    save_path = "outputs/figures/one_mto_map.png"
    plot_mto_map(
        pos=pos, z=z, coeff=coeff,
        mol_label=label, slot_idx=args.slot,
        K=K, is_synthetic=True, save_path=save_path,
    )

    print(f"=== Map generated: {save_path} ===")
    return save_path


if __name__ == "__main__":
    main()
