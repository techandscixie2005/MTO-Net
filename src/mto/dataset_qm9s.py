"""QM9S dataset parser with spectral CSV support for Stage B/C.

Stage A (mu, alpha) loading unchanged.
Stage B/C adds IR, Raman, UV from boraden CSV files.
"""
import csv, json, os, random
import torch
from torch.utils.data import Dataset, Subset


def load_qm9s_raw(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, list):
        return data
    raise ValueError("Expected list, got " + str(type(data)))


def _read_spectral_index(data_dir, task_name, max_mol_id=None):
    """Build dict mapping mol_id -> list[float] from a spectral CSV (streaming).

    Args:
        max_mol_id: if set, stop reading after this mol_id (for subset loading).
    """
    fname = {
        "ir": "ir_boraden.csv",
        "raman": "raman_boraden.csv",
        "uv": "uv_boraden.csv",
    }.get(task_name)
    if fname is None:
        raise ValueError(f"Unknown spectral task: {task_name}")
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Spectral file not found: {path}. "
            f"QM9S Figshare uses 'boraden' spelling (not 'broaden')."
        )
    index = {}
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            try:
                mol_id = int(row[0])
                if max_mol_id is not None and mol_id >= max_mol_id:
                    break
                index[mol_id] = [float(x) for x in row[1:]]
            except (ValueError, IndexError):
                continue
    return index


def _read_spectral_index_for_ids(data_dir, task_name, mol_ids):
    """Build spectral index only for specific mol_ids (efficient subset loading).

    Reads CSV streaming, stops when all requested IDs are found.
    """
    fname = {
        "ir": "ir_boraden.csv",
        "raman": "raman_boraden.csv",
        "uv": "uv_boraden.csv",
    }.get(task_name)
    if fname is None:
        raise ValueError(f"Unknown spectral task: {task_name}")
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Spectral file not found: {path}")

    mol_ids = set(mol_ids)
    index = {}
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            try:
                mol_id = int(row[0])
                if mol_id in mol_ids:
                    index[mol_id] = [float(x) for x in row[1:]]
                    if len(index) >= len(mol_ids):
                        break
            except (ValueError, IndexError):
                continue
    return index


def load_spectral_index(data_dir, tasks):
    """Build spectral indices for a list of task names.

    Returns dict[task_name] -> dict[mol_id -> list[float]]
    """
    idx = {}
    for task in tasks:
        if task in ("mu", "alpha"):
            continue
        idx[task] = _read_spectral_index(data_dir, task)
    return idx


def load_spectral_index_for_subset(data_dir, tasks, mol_ids):
    """Build spectral indices only for a subset of molecule IDs.

    Much faster than loading full index for smoke/small tests.

    Args:
        data_dir: path to qm9s data directory
        tasks: list of spectral task names
        mol_ids: set/list of molecule IDs to load

    Returns:
        dict[task_name] -> dict[mol_id -> list[float]]
    """
    idx = {}
    for task in tasks:
        if task in ("mu", "alpha"):
            continue
        idx[task] = _read_spectral_index_for_ids(data_dir, task, mol_ids)
    return idx


class QM9SDataset(Dataset):
    def __init__(self, data_list):
        self.samples = []
        for mol in data_list:
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
            self.samples.append(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SpectralCollator:
    """Collate function factory that includes spectral targets.

    Usage:
        collator = SpectralCollator(spectral_index, spectral_tasks, downsample=100)
        loader = DataLoader(dataset, collate_fn=collator)
    """

    def __init__(self, spectral_index=None, spectral_tasks=None, downsample=0):
        self.spectral_index = spectral_index or {}
        self.spectral_tasks = set(spectral_tasks or [])
        self.downsample = downsample

    def __call__(self, batch):
        """PyG-style batching. batch items may be PyG Data objects or dicts."""
        # Handle both PyG Data objects and dict samples
        z_list, pos_list, batch_idx_parts = [], [], []
        mu_list, alpha_list = [], []
        mol_ids = []

        for i, d in enumerate(batch):
            if isinstance(d, dict):
                z = d["z"]
                pos = d["pos"]
                mu = d.get("mu")
                alpha = d.get("alpha")
                mol_id = d.get("mol_id")
            else:
                z = d.z.long()
                pos = d.pos.float()
                mu = d.dipole.float().reshape(3) if hasattr(d, "dipole") else None
                alpha = d.polar.float().reshape(9) if hasattr(d, "polar") else None
                mol_id = int(d.number) if hasattr(d, "number") else None

            z_list.append(z)
            pos_list.append(pos)
            batch_idx_parts.append(torch.full((len(z),), i, dtype=torch.long))
            if mu is not None:
                mu_list.append(mu)
            if alpha is not None:
                alpha_list.append(alpha)
            mol_ids.append(mol_id)

        result = {
            "z": torch.cat(z_list),
            "pos": torch.cat(pos_list),
            "batch": torch.cat(batch_idx_parts),
        }
        if mu_list:
            result["mu"] = torch.stack(mu_list)
        if alpha_list:
            result["alpha"] = torch.stack(alpha_list)

        # Attach spectral targets by mol_id lookup
        for task in sorted(self.spectral_tasks):
            task_idx = self.spectral_index.get(task)
            if task_idx is None:
                continue
            task_list = []
            for mol_id in mol_ids:
                if mol_id is not None and mol_id in task_idx:
                    t = torch.tensor(task_idx[mol_id], dtype=torch.float32)
                    if self.downsample > 0 and len(t) > self.downsample:
                        # Downsample evenly
                        indices = torch.linspace(0, len(t) - 1, self.downsample).long()
                        t = t[indices]
                    task_list.append(t)
                else:
                    task_list.append(None)
            if all(t is not None for t in task_list):
                result[task] = torch.stack(task_list)

        return result


def collate_batch(batch):
    """PyG-style batching (backward compatible, no spectral data)."""
    collator = SpectralCollator()
    return collator(batch)


def make_split(dataset, train_frac=0.8, val_frac=0.1, seed=0):
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {"train": Subset(dataset, indices[:n_train]),
            "val": Subset(dataset, indices[n_train:n_train + n_val]),
            "test": Subset(dataset, indices[n_train + n_val:]),
            "split_seed": seed}
