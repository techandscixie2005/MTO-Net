#!/usr/bin/env python3
"""Evaluate stage transfer stability (Stage A → B → C)."""

import argparse
import json
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mto.compat import *
from src.mto.mto_model import MTONet
from src.mto.stability import extract_mto_contributions, stage_stability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--properties", nargs="+", default=["mu", "alpha"])
    parser.add_argument("--data-dir", default="data/qm9s/processed")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    test_path = os.path.join(args.data_dir, f"{args.split}.pt")
    if not os.path.exists(test_path):
        z = torch.randint(1, 9, (20,)).to(device)
        pos = torch.randn(20, 3).to(device)
        batch = torch.zeros(20, dtype=torch.long).to(device)
        test_input = {"z": z, "pos": pos, "batch": batch}
    else:
        test_data = torch.load(test_path, map_location="cpu", weights_only=False)[:4]
        z = torch.cat([s["z"] for s in test_data]).to(device)
        pos = torch.cat([s["pos"] for s in test_data]).to(device)
        batch = torch.zeros(len(z), dtype=torch.long).to(device)
        test_input = {"z": z, "pos": pos, "batch": batch}

    stages = ["stage_a", "stage_b", "stage_c"]
    contribs = {}
    for stage in stages:
        ckpt_path = os.path.join(args.checkpoint_dir, f"{stage}_seed0", "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"SKIP: {ckpt_path}")
            continue
        print(f"Loading {stage} from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        tasks = ckpt.get("tasks", {k: 3 for k in ["mu", "alpha"]})
        model = MTONet(tasks=tasks).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        try:
            contribs[stage] = extract_mto_contributions(
                model, test_input["z"], test_input["pos"], test_input["batch"], top_r=5)
        except Exception as e:
            print(f"  ERROR: {e}")

    if len(contribs) < 2:
        print("Need at least 2 stages")
        return

    results = []
    keys = sorted(contribs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            result = stage_stability(contribs[keys[i]], contribs[keys[j]])
            result["stage_a"] = keys[i]
            result["stage_b"] = keys[j]
            results.append(result)
            print(f"  {keys[i]} → {keys[j]}: corr={result[mean_correlation]:.4f}")

    os.makedirs(f"{args.out_dir}/metrics", exist_ok=True)
    with open(f"{args.out_dir}/metrics/stage_stability.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {args.out_dir}/metrics/stage_stability.json")


if __name__ == "__main__":
    main()
