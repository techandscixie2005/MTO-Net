"""QM9S dataset parser for PyTorch / PyG Data format.

The QM9S dataset is stored as a single torch_geometric file (qm9s.pt).
This module loads, wraps, and splits it for MTO-Net training.
"""

import json
import os
import random
from typing import Optional

import torch
from torch.utils.data import Dataset, Subset


def load_qm9s_raw(path: str) -> list:
    """Load raw QM9S data from a .pt file. Returns list of Data objects."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a list of Data objects, got {type(data)}")


class QM9SDataset(Dataset):
    """Wraps a list of QM9S molecules as a PyTorch Dataset.

    Each item is a dict with keys:
        z, pos, smiles (if available), mol_id,
        mu, alpha, ir, raman, uv, quadrupole, octupole, etc.
    Field names adapt to the actual qm9s.pt structure.
    """

    PREFERRED_KEYS = [
        "z", "pos", "smiles", "mol_id",
        "mu", "alpha", "ir", "raman", "uv",
        "quadrupole", "octupole",
        "homo", "lumo", "gap",
        "energy", "forces",
    ]

    def __init__(self, data_list: list):
        self.samples = []
        for i, mol in enumerate(data_list):
            sample = self._extract_fields(mol, i)
            self.samples.append(sample)

    def _extract_fields(self, mol, idx: int) -> dict:
        sample = {"mol_id": idx}

        if hasattr(mol, "z"):
            sample["z"] = torch.as_tensor(mol.z, dtype=torch.long)
        elif hasattr(mol, "atomic_numbers"):
            sample["z"] = torch.as_tensor(mol.atomic_numbers, dtype=torch.long)

        if hasattr(mol, "pos"):
            sample["pos"] = torch.as_tensor(mol.pos, dtype=torch.float32)
        elif hasattr(mol, "positions"):
            sample["pos"] = torch.as_tensor(mol.positions, dtype=torch.float32)

        if hasattr(mol, "smiles"):
            sample["smiles"] = str(mol.smiles)
        elif hasattr(mol, "smi"):
            sample["smiles"] = str(mol.smi)

        # Dipole moment
        if hasattr(mol, "mu"):
            sample["mu"] = torch.as_tensor(mol.mu, dtype=torch.float32)
        elif hasattr(mol, "dipole"):
            sample["mu"] = torch.as_tensor(mol.dipole, dtype=torch.float32)

        # Polarizability
        for attr in ["alpha", "polarizability", "pol"]:
            if hasattr(mol, attr):
                v = getattr(mol, attr)
                v = torch.as_tensor(v, dtype=torch.float32)
                if v.dim() == 1 and v.shape[0] == 6:
                    # Upper triangular -> 3x3 symmetric matrix
                    tri = v
                    mat = torch.zeros(3, 3)
                    mat[0, 0] = tri[0]; mat[0, 1] = tri[1]; mat[0, 2] = tri[3]
                    mat[1, 0] = tri[1]; mat[1, 1] = tri[2]; mat[1, 2] = tri[4]
                    mat[2, 0] = tri[3]; mat[2, 1] = tri[4]; mat[2, 2] = tri[5]
                    v = mat
                sample["alpha"] = v
                break

        # Spectra
        for attr, key in [("ir", "ir"), ("raman", "raman"), ("uv", "uv"),
                          ("ir_spectrum", "ir"), ("raman_spectrum", "raman"),
                          ("uv_spectrum", "uv"), ("uvvis", "uv")]:
            if hasattr(mol, attr):
                sample[key] = torch.as_tensor(getattr(mol, attr), dtype=torch.float32)

        # Quadrupole / Octupole
        for attr, key in [("quadrupole", "quadrupole"), ("octupole", "octupole")]:
            if hasattr(mol, attr):
                sample[key] = torch.as_tensor(getattr(mol, attr), dtype=torch.float32)

        # HOMO / LUMO / gap
        for attr, key in [("homo", "homo"), ("lumo", "lumo"), ("gap", "gap"),
                          ("energy", "energy")]:
            if hasattr(mol, attr):
                v = getattr(mol, attr)
                sample[key] = torch.as_tensor(v, dtype=torch.float32).reshape(-1)

        return sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_split(dataset, train_frac=0.8, val_frac=0.1, seed=0):
    """Deterministic train/val/test split."""
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train": Subset(dataset, indices[:n_train]),
        "val": Subset(dataset, indices[n_train:n_train + n_val]),
        "test": Subset(dataset, indices[n_train + n_val:]),
        "split_seed": seed,
    }


def collate_fn(batch: list) -> dict:
    """Collate a list of sample dicts into a batched dict."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        vals = [s[k] for s in batch if k in s]
        if not vals:
            continue
        if isinstance(vals[0], torch.Tensor):
            try:
                result[k] = torch.stack(vals)
            except RuntimeError:
                result[k] = vals
        elif isinstance(vals[0], str):
            result[k] = vals
        else:
            result[k] = vals
    return result
