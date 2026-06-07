import torch
import pytest
from src.mto.valence import molecular_valence_electrons, VALENCE_ELECTRONS


class TestValenceElectrons:
    def test_known_elements(self):
        assert VALENCE_ELECTRONS[1] == 1   # H
        assert VALENCE_ELECTRONS[6] == 4   # C
        assert VALENCE_ELECTRONS[7] == 5   # N
        assert VALENCE_ELECTRONS[8] == 6   # O
        assert VALENCE_ELECTRONS[9] == 7   # F

    def test_single_molecule(self):
        z = torch.tensor([6, 8, 6])  # C, O, C
        batch = torch.zeros(3, dtype=torch.long)
        K = molecular_valence_electrons(z, batch)
        assert K.tolist() == [4 + 6 + 4]

    def test_batch_of_molecules(self):
        z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1])
        batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
        K = molecular_valence_electrons(z, batch)
        assert K.tolist() == [4 + 4 + 6 + 6 * 1]

    def test_variable_K(self):
        z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1, 8, 1, 1])
        batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        K = molecular_valence_electrons(z, batch)
        assert K[0].item() == 4 + 4 + 6 + 6  # C2H6O = 20
        assert K[1].item() == 6 + 2           # OH2 = 8

    def test_unknown_element_raises(self):
        z = torch.tensor([6, 92, 8])  # U (uranium)
        batch = torch.zeros(3, dtype=torch.long)
        with pytest.raises(ValueError, match="Unknown atomic numbers"):
            molecular_valence_electrons(z, batch)

    def test_hydrogen(self):
        z = torch.tensor([1, 1])
        batch = torch.zeros(2, dtype=torch.long)
        K = molecular_valence_electrons(z, batch)
        assert K.tolist() == [2]


class TestValencePadding:
    def test_K_max_padding(self):
        z1 = torch.tensor([1, 1])           # K=2
        z2 = torch.tensor([6, 8, 8, 6, 6])  # K=4+6+6+4+4=24
        z = torch.cat([z1, z2])
        batch = torch.cat([torch.zeros(2, dtype=torch.long),
                           torch.ones(5, dtype=torch.long)])
        K = molecular_valence_electrons(z, batch)
        assert K[0].item() == 2
        assert K[1].item() == 24  # C3O2: 3*4+2*6=24
        assert int(K.max().item()) > int(K.min().item())
