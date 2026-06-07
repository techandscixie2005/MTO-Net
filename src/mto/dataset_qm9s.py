"""QM9S dataset parser matching actual qm9s.pt fields."""
import json, os, random
import torch
from torch.utils.data import Dataset, Subset

def load_qm9s_raw(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, list):
        return data
    raise ValueError("Expected list, got " + str(type(data)))

class QM9SDataset(Dataset):
    """Wraps PyG Data objects from qm9s.pt.

    Actual QM9S fields in qm9s.pt:
      z, pos, smile, number, dipole, polar, energy, npacharge,
      quadrupole, octapole, hyperpolar, tran_energy, tran_dipole,
      dedipole, depolar, Hi, Hij, edge_index
    NO ir/raman/uv in .pt - those are in separate CSV files.
    """
    def __init__(self, data_list):
        self.samples = []
        for mol in data_list:
            s = self._extract(mol)
            self.samples.append(s)

    def _extract(self, mol):
        s = {}
        # Core atoms
        s["z"] = mol.z.long()
        s["pos"] = mol.pos.float()
        if hasattr(mol, "smile"):
            s["smiles"] = str(mol.smile)
        if hasattr(mol, "number"):
            s["mol_id"] = int(mol.number)

        # Build batch index (for batching within mol)
        s["batch"] = torch.zeros(len(s["z"]), dtype=torch.long)

        # Dipole: [1,3] -> [3]
        if hasattr(mol, "dipole"):
            d = mol.dipole.float()
            s["mu"] = d.reshape(-1)

        # Polar: [1,3,3] -> [9] (flattened)
        if hasattr(mol, "polar"):
            p = mol.polar.float()
            s["alpha"] = p.reshape(-1)

        # Energy
        if hasattr(mol, "energy"):
            s["energy"] = mol.energy.float().reshape(-1)

        # Quadrupole, octupole, hyperpolar (for future use)
        for key in ["quadrupole", "octapole", "hyperpolar"]:
            if hasattr(mol, key):
                s[key] = getattr(mol, key).float().reshape(-1)

        # Charges
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

def collate_fn(batch):
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
