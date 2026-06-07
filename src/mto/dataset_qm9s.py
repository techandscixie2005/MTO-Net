"""QM9S dataset parser."""
import json, os, random
from typing import Optional
import torch
from torch.utils.data import Dataset, Subset

def load_qm9s_raw(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected list, got {type(data)}")

class QM9SDataset(Dataset):
    """Wraps PyG Data objects from qm9s.pt.
    
    Exact field mapping from QM9S:
      - z: [N] int64 (atomic numbers)
      - pos: [N,3] float32
      - smile: str (SMILES)
      - number: int (molecule number)
      - dipole: [1,3] -> mu
      - polar: [1,3,3] -> alpha
      - npacharge: [N] -> charges
      - energy: [1,1]
      - quadrupole, octapole, hyperpolar
      - tran_energy, tran_dipole (excitation)
      - dedipole, depolar (derivatives)
      - Hi, Hij (Hessian parts)
    """

    FIELD_MAP = {
        "dipole": "mu",
        "polar": "alpha",
        "npacharge": "charges",
        "smile": "smiles",
        "number": "mol_id",
    }

    def __init__(self, data_list):
        self.samples = []
        for mol in data_list:
            sample = self._extract(mol)
            self.samples.append(sample)

    def _extract(self, mol):
        s = {}
        for key in mol.keys():
            v = mol[key]
            out_key = self.FIELD_MAP.get(key, key)
            if isinstance(v, torch.Tensor):
                if v.dtype in (torch.float32, torch.float64):
                    s[out_key] = v.float().squeeze(0)
                else:
                    s[out_key] = v
            elif isinstance(v, str):
                s[out_key] = v
            elif isinstance(v, (int, float)):
                s[out_key] = v
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
