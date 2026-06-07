"""Tests for QM9S dataset parser."""
import torch
from unittest.mock import MagicMock

def test_dataset_extract_fields():
    from src.mto.dataset_qm9s import QM9SDataset
    mol = MagicMock()
    mol.z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1])
    mol.pos = torch.randn(9, 3)
    mol.smile = "CCO"
    mol.number = 42
    mol.dipole = torch.tensor([0.1, 0.2, 0.3]).unsqueeze(0)
    mol.polar = torch.eye(3).unsqueeze(0)
    ds = QM9SDataset([mol])
    assert len(ds) == 1
    sample = ds[0]
    assert "z" in sample
    assert "smiles" in sample
    assert "mu" in sample
    assert "alpha" in sample

def test_dataset_alpha_shapes():
    from src.mto.dataset_qm9s import QM9SDataset
    mol = MagicMock()
    mol.z = torch.tensor([6])
    mol.pos = torch.randn(1, 3)
    mol.polar = torch.eye(3).unsqueeze(0)
    ds = QM9SDataset([mol])
    alpha = ds[0]["alpha"]
    assert list(alpha.shape) == [9]

def test_collate_batch():
    from src.mto.dataset_qm9s import collate_batch
    mol1 = MagicMock()
    mol1.z = torch.tensor([6, 8, 1, 1])
    mol1.pos = torch.randn(4, 3)
    mol1.dipole = torch.tensor([0.1, 0.2, 0.3]).unsqueeze(0)
    mol1.polar = torch.eye(3).unsqueeze(0)
    mol2 = MagicMock()
    mol2.z = torch.tensor([6, 1, 1, 1, 1])
    mol2.pos = torch.randn(5, 3)
    mol2.dipole = torch.tensor([0.4, 0.5, 0.6]).unsqueeze(0)
    mol2.polar = torch.eye(3).unsqueeze(0)
    result = collate_batch([mol1, mol2])
    assert result["z"].shape == (9,)
    assert result["pos"].shape == (9, 3)
    assert result["mu"].shape == (2, 3)
    assert result["batch"].shape == (9,)

def test_make_split():
    from src.mto.dataset_qm9s import QM9SDataset, make_split
    mols = []
    for i in range(100):
        mol = MagicMock()
        mol.z = torch.randint(1, 10, (5,)).clamp(1, 9)
        mol.pos = torch.randn(5, 3)
        mols.append(mol)
    ds = QM9SDataset(mols)
    splits = make_split(ds, train_frac=0.8, val_frac=0.1, seed=42)
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10
