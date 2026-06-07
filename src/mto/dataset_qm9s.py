"""QM9S dataset parser."""
import json, os, random
import torch
from torch.utils.data import Dataset, Subset

def load_qm9s_raw(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, list):
        return data
    raise ValueError("Expected list, got " + str(type(data)))

class QM9SDataset(Dataset):
    def __init__(self, data_list):
        self.samples = []
        for mol in data_list:
            s = self._extract(mol)
            self.samples.append(s)

    def _extract(self, mol):
        s = {}
        s["z"] = mol.z.long()
        s["pos"] = mol.pos.float()
        if hasattr(mol, "smile"):
            s["smiles"] = str(mol.smile)
        if hasattr(mol, "number"):
            s["mol_id"] = int(mol.number)
        if hasattr(mol, "dipole"):
            s["mu"] = mol.dipole.float().reshape(3)
        if hasattr(mol, "polar"):
            s["alpha"] = mol.polar.float().reshape(9)
        if hasattr(mol, "energy"):
            s["energy"] = mol.energy.float().reshape(-1)
        for key in ["quadrupole", "octapole", "hyperpolar"]:
            if hasattr(mol, key):
                s[key] = getattr(mol, key).float().reshape(-1)
        if hasattr(mol, "npacharge"):
            s["charges"] = mol.npacharge.float()
        return s

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_split(dataset, train_frac=0.8, val_frac=0.1, seed=0):
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


def collate_batch(batch):
    """PyG-style batching: concat all atoms, build batch index, stack labels."""
    z_all = torch.cat([s["z"] for s in batch])
    pos_all = torch.cat([s["pos"] for s in batch])
    batch_idx = torch.cat([torch.full((len(s["z"]),), i, dtype=torch.long) for i, s in enumerate(batch)])

    result = {"z": z_all, "pos": pos_all, "batch": batch_idx}

    for key in batch[0]:
        if key in ("z", "pos", "batch"):
            continue
        vals = [s[key] for s in batch if key in s]
        if not vals:
            continue
        if isinstance(vals[0], torch.Tensor):
            try:
                result[key] = torch.stack(vals)
            except RuntimeError:
                result[key] = vals
        else:
            result[key] = vals

    return result
