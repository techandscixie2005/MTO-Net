import torch
import pytest
from src.mto.mto_module import ValenceAdaptiveMTO
from src.mto.valence import molecular_valence_electrons


class TestMTOShapes:
    def setup_method(self):
        self.C = 128
        self.mto = ValenceAdaptiveMTO(feature_dim=self.C)

    def test_single_molecule_output_shapes(self):
        z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1])
        pos = torch.randn(9, 3)
        batch = torch.zeros(9, dtype=torch.long)
        atom_features = torch.randn(9, self.C)

        out = self.mto(atom_features=atom_features, z=z, batch=batch)

        K = int(molecular_valence_electrons(z, batch)[0])
        B = 1
        assert out["O"].shape == (B, K, self.C), f"O shape {out['O'].shape}"
        assert out["coeff"].shape[0] == B
        assert out["coeff"].shape[1] == K
        assert out["mask"].shape == (B, K)
        assert out["atom_mask"].shape == (B, 9)
        assert out["K_per_mol"].tolist() == [K]

    def test_batch_output_shapes(self):
        z1 = torch.tensor([6, 8, 1, 1, 1, 1])    # CH3OH, K=4+6+4=14
        z2 = torch.tensor([1, 1])                   # H2, K=2
        z = torch.cat([z1, z2])
        batch = torch.cat([torch.zeros(6, dtype=torch.long),
                           torch.ones(2, dtype=torch.long)])
        atom_features = torch.randn(8, self.C)

        out = self.mto(atom_features=atom_features, z=z, batch=batch)

        K0 = int(molecular_valence_electrons(z1, torch.zeros(6, dtype=torch.long))[0])
        K1 = int(molecular_valence_electrons(z2, torch.zeros(2, dtype=torch.long))[0])
        K_max = max(K0, K1)
        B = 2
        N_max = 6

        assert out["O"].shape == (B, K_max, self.C)
        assert out["coeff"].shape == (B, K_max, N_max)
        assert out["mask"].shape == (B, K_max)
        assert out["atom_mask"].shape == (B, N_max)

    def test_no_nan_in_forward(self):
        torch.manual_seed(42)
        z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1])
        batch = torch.zeros(9, dtype=torch.long)
        atom_features = torch.randn(9, self.C)

        out = self.mto(atom_features=atom_features, z=z, batch=batch)

        assert not torch.isnan(out["O"]).any()
        assert not torch.isinf(out["O"]).any()
        assert not torch.isnan(out["coeff"]).any()
        assert not torch.isinf(out["coeff"]).any()

    def test_coeff_normalization(self):
        torch.manual_seed(123)
        z = torch.tensor([6, 8, 1, 1, 1, 1])
        batch = torch.zeros(6, dtype=torch.long)
        atom_features = torch.randn(6, self.C)

        out = self.mto(atom_features=atom_features, z=z, batch=batch)

        K = int(out["K_per_mol"][0])
        coeff = out["coeff"][0, :K, :6]
        abs_sum = coeff.abs().sum(dim=-1)
        assert torch.allclose(abs_sum, torch.ones(K), atol=1e-5),             f"abs_sum={abs_sum}"

    def test_variable_K_padding(self):
        z1 = torch.tensor([1, 1])               # K=2
        z2 = torch.tensor([6, 6, 8, 6, 6, 6])   # K=4*5+6=26
        z = torch.cat([z1, z2])
        batch = torch.cat([torch.zeros(2, dtype=torch.long),
                           torch.ones(6, dtype=torch.long)])
        atom_features = torch.randn(8, self.C)

        out = self.mto(atom_features=atom_features, z=z, batch=batch)

        K0 = int(out["K_per_mol"][0])
        K1 = int(out["K_per_mol"][1])
        assert K0 == 2
        assert K1 == 4 * 5 + 6  # 26
        K_max = max(K0, K1)

        # Molecule 0 only has 2 valid slots; slots 2..K_max-1 masked
        assert out["mask"][0, :K0].all()
        if K0 < K_max:
            assert not out["mask"][0, K0:].any()

        # Molecule 0 only has 2 atoms
        assert out["atom_mask"][0, :2].all()
        if out["atom_mask"].shape[1] > 2:
            assert not out["atom_mask"][0, 2:].any()
