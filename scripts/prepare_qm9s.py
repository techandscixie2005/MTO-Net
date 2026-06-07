#!/usr/bin/env python3
"""Parse QM9S raw data and create processed PyTorch datasets."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mto.dataset_qm9s import load_qm9s_raw, QM9SDataset, make_split


def inspect_sample(sample, idx=0):
    """Print available fields in first sample."""
    print(f"\nSample {idx} fields:")
    for key in sorted(sample.keys()):
        val = sample[key]
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={list(val.shape)}, dtype={val.dtype}")
        elif isinstance(val, str):
            print(f"  {key}: str={val[:80]}")
        else:
            print(f"  {key}: {type(val).__name__}={val}")
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/qm9s")
    parser.add_argument("--out", default="data/qm9s/processed")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit molecules (0 = all)")
    args = parser.parse_args()

    pt_path = os.path.join(args.data_dir, "qm9s.pt")
    if not os.path.exists(pt_path):
        print(f"ERROR: qm9s.pt not found at {pt_path}")
        print("Download QM9S first: python scripts/download_qm9s.py")
        sys.exit(1)

    print(f"Loading QM9S from {pt_path} ...")
    raw_data = load_qm9s_raw(pt_path)
    print(f"Loaded {len(raw_data)} molecules")

    if args.limit > 0:
        raw_data = raw_data[:args.limit]
        print(f"Limited to {args.limit} molecules")

    # Inspect first sample
    first_mol = raw_data[0]
    print(f"First molecule type: {type(first_mol).__name__}")
    for attr in dir(first_mol):
        if not attr.startswith("_"):
            v = getattr(first_mol, attr)
            if not callable(v):
                print(f"  .{attr}: {type(v).__name__}", end="")
                if isinstance(v, (list, tuple)):
                    print(f" len={len(v)}")
                elif hasattr(v, "shape"):
                    print(f" shape={list(v.shape)}")
                else:
                    print()

    # Build dataset
    dataset = QM9SDataset(raw_data)
    print(f"\nDataset: {len(dataset)} samples")

    sample = dataset[0]
    inspect_sample(sample)

    # Split
    splits = make_split(dataset, train_frac=args.train_frac,
                        val_frac=args.val_frac, seed=args.seed)
    print(f"\nSplit (seed={args.seed}):")
    print(f"  train: {len(splits[train])}")
    print(f"  val:   {len(splits[val])}")
    print(f"  test:  {len(splits[test])}")

    # Save
    os.makedirs(args.out, exist_ok=True)
    for name, subset in splits.items():
        if name == "split_seed":
            continue
        path = os.path.join(args.out, f"{name}.pt")
        torch.save([dataset[i] for i in subset.indices], path)
        print(f"Saved {path} ({len(subset)} samples)")

    # Save split info
    info = {
        "total_mols": len(dataset),
        "split_seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
    }
    with open(os.path.join(args.out, "split_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nSplit info saved")

    # Save field manifest
    manifest = {
        "fields": {k: {"shape": list(v.shape) if isinstance(v, torch.Tensor) else str(type(v).__name__),
                       "dtype": str(v.dtype) if isinstance(v, torch.Tensor) else ""}
                   for k, v in dataset[0].items()},
    }
    with open(os.path.join(args.out, "field_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
