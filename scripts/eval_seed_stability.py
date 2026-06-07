#!/usr/bin/env python3
"""Evaluate seed subspace stability for trained MTO-Net checkpoints."""

import argparse
import json
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mto.compat import *
from src.mto.mto_model import MTONet
from src.mto.stability import extract_mto_contributions, seed_subspace_stability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="stage_a")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-r", type=int, default=5)
    parser.add_argument("--data-dir", default="data/qm9s/processed")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    test_path = os.path.join(args.data_dir, f"{args.split}.pt")
    if not os.path.exists(test_path):
        print(f"WARNING: {test_path} not found. Using synthetic data.")
        z = torch.randint(1, 9, (20,)).to(device)
        pos = torch.randn(20, 3).to(device)
        batch = torch.zeros(20, dtype=torch.long).to(device)
        test_data = [{"z": z, "pos": pos, "batch": batch}]
    else:
        test_data = torch.load(test_path, map_location="cpu", weights_only=False)

    # Load checkpoints for each seed
    contrib_maps = {}
    for seed in [0, 1, 2]:
        ckpt_path = os.path.join(args.checkpoint_dir, f"{args.stage}_seed{seed}", "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"SKIP: {ckpt_path} not found")
            continue

        print(f"Loading seed {seed}: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        tasks = ckpt.get("tasks", {k: 3 for k in ["mu", "alpha"]})
        model = MTONet(tasks=tasks).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Extract contributions for a batch
        if isinstance(test_data, list):
            # Use first few molecules
            sample = test_data[0]
            z = sample["z"].to(device)
            pos = sample["pos"].to(device)
            batch_t = sample.get("batch", torch.zeros(len(z), dtype=torch.long)).to(device)
        else:
            z = torch.cat([s["z"] for s in test_data[:8]]).to(device)
            pos = torch.cat([s["pos"] for s in test_data[:8]]).to(device)
            batch_t = torch.zeros(len(z), dtype=torch.long).to(device)

        try:
            maps = extract_mto_contributions(model, z, pos, batch_t, top_r=args.top_r)
            contrib_maps[f"seed_{seed}"] = maps
        except Exception as e:
            print(f"  ERROR extracting contributions for seed {seed}: {e}")

    if len(contrib_maps) < 2:
        print("Need at least 2 seeds for stability analysis")
        return

    # Compute stability
    result = seed_subspace_stability(contrib_maps, top_r=args.top_r)

    # Save
    os.makedirs(f"{args.out_dir}/metrics", exist_ok=True)
    os.makedirs(f"{args.out_dir}/figures/stability", exist_ok=True)

    csv_path = f"{args.out_dir}/metrics/seed_subspace_stability.csv"
    with open(csv_path, "w") as f:
        f.write("seed_a,seed_b,similarity,n_mols\n")
        for pair in result["pairs"]:
            f.write(f"{pair[seed_a]},{pair[seed_b]},{pair[similarity]:.4f},{pair[n_mols]}\n")

    json_path = f"{args.out_dir}/metrics/seed_subspace_stability.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSeed subspace stability (r={args.top_r}):")
    for pair in result["pairs"]:
        print(f"  {pair[seed_a]} vs {pair[seed_b]}: {pair[similarity]:.4f}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
