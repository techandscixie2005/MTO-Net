"""Locked dataset split: generate once, reuse across seeds.

Ensures all training seeds use the same train/val/test molecule membership.
Seed randomness only affects model init, dataloader shuffle, and training.
"""
import os
import json
import hashlib
import random


def generate_split(n_total, train_frac=0.8, val_frac=0.1, seed=0):
    """Generate train/val/test indices deterministically.

    Returns dict with keys: train, val, test, split_seed, n_total.
    """
    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    return {
        "train": sorted(indices[:n_train]),
        "val": sorted(indices[n_train:n_train + n_val]),
        "test": sorted(indices[n_train + n_val:]),
        "split_seed": seed,
        "n_total": n_total,
        "train_frac": train_frac,
        "val_frac": val_frac,
    }


def save_split(split, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
    # Save hash alongside
    hash_path = path.replace(".json", "_hash.txt")
    content = json.dumps(split, sort_keys=True)
    h = hashlib.sha256(content.encode()).hexdigest()[:16]
    with open(hash_path, "w") as f:
        f.write(h + "\n")
    return h


def load_split(path):
    with open(path) as f:
        split = json.load(f)
    required = {"train", "val", "test", "split_seed", "n_total"}
    missing = required - set(split.keys())
    if missing:
        raise KeyError(f"Split file missing keys: {missing}")
    return split


def load_or_create_split(path, n_total, train_frac=0.8, val_frac=0.1, seed=0):
    """Load split from file, or create and save if missing."""
    if os.path.exists(path):
        split = load_split(path)
        if split["n_total"] != n_total:
            raise ValueError(
                f"Dataset size changed: split has {split['n_total']}, "
                f"current dataset has {n_total}. Regenerate the split."
            )
        return split, False  # loaded
    split = generate_split(n_total, train_frac, val_frac, seed)
    save_split(split, path)
    return split, True  # created


def split_indices_for_seed(split, seed):
    """Return (train_indices, val_indices, test_indices).

    Dataset membership is locked by the split file.
    Seed only affects the shuffle order within each set (for DataLoader).
    """
    return split["train"], split["val"], split["test"]
