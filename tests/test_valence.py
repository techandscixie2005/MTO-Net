"""Tests for valence electron counting."""

import torch
import pytest


def test_valence_hydrogen():
    from src.mto.valence import molecular_valence_electrons, VALENCE_ELECTRONS
    assert VALENCE_ELECTRONS[1] == 1


def test_valence_ethanol():
    from src.mto.valence import molecular_valence_electrons
    # Ethanol: C2H6O = 2*4 + 6*1 + 6 = 20
    z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    batch = torch.zeros(9, dtype=torch.long)
    K = molecular_valence_electrons(z, batch)
    assert K.item() == 20


def test_valence_batch():
    from src.mto.valence import molecular_valence_electrons
    # Molecule 1: H2O (2*1+6=8), Molecule 2: CH4 (4+4*1=8)
    z = torch.tensor([8, 1, 1, 6, 1, 1, 1, 1], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    K = molecular_valence_electrons(z, batch)
    assert int(K[0]) == 8  # H2O
    assert int(K[1]) == 8  # CH4


def test_valence_unknown_z():
    from src.mto.valence import molecular_valence_electrons
    z = torch.tensor([6, 8, 99], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError):
        molecular_valence_electrons(z, batch)
