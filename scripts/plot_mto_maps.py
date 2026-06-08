#!/usr/bin/env python3
"""Plot MTO atom contribution maps from a trained checkpoint."""

import argparse
import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mto.compat import *
from src.mto.mto_model import MTONet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-mols", type=int, default=4)
    parser.add_argument("--top-slots", type=int, default=3)
    parser.add_argument("--out-dir", default="outputs/figures/mto_maps")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    tasks = ckpt.get("tasks", {k: 3 for k in ["mu", "alpha"]})
    model = MTONet(tasks=tasks).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Generating MTO maps for up to {args.num_mols} molecules...")
    figures = []

    for mol_idx in range(min(args.num_mols, 8)):
        # Generate a synthetic molecule for visualization
        # In practice this would use real QM9S data
        n_atoms = np.random.randint(5, 20)
        z = torch.randint(1, 9, (n_atoms,)).to(device)
        pos = torch.randn(n_atoms, 3).to(device)
        batch = torch.zeros(n_atoms, dtype=torch.long).to(device)

        with torch.no_grad():
            out = model(z=z, pos=pos, batch=batch, return_mto=True)

        coeff = out["coeff"][0]  # [K_max, N_max]
        K_per_mol = int(out["K_per_mol"][0])

        n_valid = min(args.top_slots, K_per_mol)
        if n_valid == 0:
            continue

        fig, axes = plt.subplots(1, n_valid, figsize=(3 * n_valid, 3))
        if n_valid == 1:
            axes = [axes]

        for k in range(n_valid):
            c = coeff[k, :n_atoms].cpu().numpy()
            im = axes[k].bar(range(n_atoms), c if c.sum() > 0 else np.abs(c),
                           color="steelblue", edgecolor="navy")
            axes[k].set_title(f"Slot {k}")
            axes[k].set_xlabel("Atom index")
            if k == 0:
                axes[k].set_ylabel("Contribution")

        fig.suptitle(f"Molecule {mol_idx} (K={K_per_mol}, atoms={n_atoms})",
                    fontsize=14, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(args.out_dir, f"smoke_mol{mol_idx}_mto.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path)
        print(f"  Saved: {path}")

    print(f"\nGenerated {len(figures)} MTO maps")


if __name__ == "__main__":
    main()
