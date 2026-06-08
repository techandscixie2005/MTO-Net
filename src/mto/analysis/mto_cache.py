"""MTO interpretability cache: save/load MTO activations and contributions."""
import os
import json
import torch
import numpy as np


DEFAULT_CACHE_FIELDS = [
    "mol_id", "smiles", "z", "pos", "K", "mto_mask",
    "routing_attention", "signed_coeff", "activity",
    "mto_tensor", "atom_contrib_norm",
    "pred", "target",
    "seed", "stage", "checkpoint",
]


class MTOCache:
    """Size-controlled MTO interpretability cache."""

    def __init__(self, save_dir, max_molecules=512, compress=True):
        self.save_dir = save_dir
        self.max_molecules = max_molecules
        self.compress = compress
        self._entries = []
        os.makedirs(save_dir, exist_ok=True)

    def add_entry(self, entry):
        entry = {k: self._to_serializable(v) for k, v in entry.items()}
        self._entries.append(entry)

    def save(self, epoch=None, tag="best"):
        if len(self._entries) > self.max_molecules:
            self._entries = self._entries[:self.max_molecules]

        ext = ".npz" if self.compress else ".npy"
        suffix = f"_epoch{epoch}_{tag}" if epoch is not None else f"_{tag}"
        fname = f"mto_cache{suffix}{ext}"
        path = os.path.join(self.save_dir, fname)

        arrays = {}
        meta = []
        for i, entry in enumerate(self._entries):
            meta.append({k: v for k, v in entry.items()
                        if not isinstance(v, np.ndarray)})
            for k, v in entry.items():
                if isinstance(v, np.ndarray):
                    arrays[f"{i}_{k}"] = v

        np.savez_compressed(path, **arrays) if self.compress else np.save(path, arrays)
        meta_path = path.replace(ext, "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return path

    def clear(self):
        self._entries = []

    def __len__(self):
        return len(self._entries)

    @staticmethod
    def _to_serializable(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, (int, float, str, bool)):
            return value
        if isinstance(value, list):
            return [MTOCache._to_serializable(v) for v in value]
        if isinstance(value, np.ndarray):
            return value
        return str(value)


def build_mto_cache_entry(model_output, batch, mol_idx=0,
                          seed=0, stage="stage_a", checkpoint="best"):
    """Build a cache entry from model output and batch."""
    z = batch["z"]
    batch_idx = batch.get("batch", torch.zeros(len(z), dtype=torch.long))
    mask_mol = batch_idx == mol_idx
    z_mol = z[mask_mol]

    K = model_output.get("K_per_mol", None)
    K_mol = int(K[mol_idx].item()) if K is not None else 0

    coeff = model_output.get("coeff", None)
    activity = model_output.get("activity", None)

    # Atom contribution norms
    if coeff is not None:
        coeff_mol = coeff[mol_idx]  # [K_mol, N]
        atom_contrib = (coeff_mol.abs().sum(dim=0) / max(coeff_mol.shape[0], 1))
        atom_contrib_norm = atom_contrib.detach().cpu().numpy()
    else:
        atom_contrib_norm = np.zeros(len(z_mol))

    entry = {
        "mol_id": int(mol_idx),
        "z": z_mol,
        "K": K_mol,
        "mto_mask": model_output.get("mask", torch.zeros(1, 1))[mol_idx].detach().cpu().numpy(),
        "signed_coeff": coeff[mol_idx].detach().cpu().numpy() if coeff is not None else np.zeros((1, 1)),
        "routing_attention": np.zeros((1, 1)),
        "activity": activity[mol_idx].detach().cpu().numpy() if activity is not None else np.ones(K_mol),
        "mto_tensor": model_output.get("O", torch.zeros(1, 1, 1))[mol_idx].detach().cpu().numpy(),
        "atom_contrib_norm": atom_contrib_norm,
        "seed": seed,
        "stage": stage,
        "checkpoint": checkpoint,
    }

    # Predictions and targets
    for key in ("mu", "alpha"):
        if key in model_output:
            entry[f"pred_{key}"] = model_output[key][mol_idx].detach().cpu().numpy()
        if key in batch:
            entry[f"target_{key}"] = batch[key][mol_idx].detach().cpu().numpy()

    return entry
