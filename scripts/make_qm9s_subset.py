#!/usr/bin/env python3
"""Create small subsets of QM9S for smoke testing and medium runs."""

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="smoke", help="Subset name")
    parser.add_argument("--num-mols", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="data/qm9s")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(args.data_dir, f"subset_{args.name}")

    source_path = os.path.join(args.data_dir, "qm9s.pt")
    if not os.path.exists(source_path):
        print(f"ERROR: qm9s.pt not found at {source_path}")
        print("Download first: python scripts/download_qm9s.py")
        sys.exit(1)

    print(f"Loading QM9S from {source_path} ...")
    data = torch.load(source_path, map_location="cpu", weights_only=False)
    print(f"Total molecules available: {len(data)}")

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(data)), min(args.num_mols, len(data)))
    subset = [data[i] for i in sorted(indices)]

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "qm9s.pt")
    torch.save(subset, out_path)
    print(f"Saved {len(subset)} molecules to {out_path}")

    # Also save index list
    with open(os.path.join(args.out, "indices.txt"), "w") as f:
        for idx in sorted(indices):
            f.write(f"{idx}\n")

    info = {
        "name": args.name,
        "num_mols": len(subset),
        "seed": args.seed,
        "source": source_path,
        "indices": sorted(indices),
    }
    import json
    with open(os.path.join(args.out, "subset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"Subset info saved")


if __name__ == "__main__":
    main()
