"""Valence-adaptive MTO module with batched, per-molecule slot routing."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .valence import molecular_valence_electrons


class ValenceAdaptiveMTO(nn.Module):
    """Valence-adaptive MTO: K = total valence electrons per molecule.

    Operates molecule-by-molecule within a batch (variable N, variable K).
    This is naturally sequential per molecule but the inner routing is vectorized.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        num_types = 20
        self.atom_type_emb = nn.Embedding(num_types, hidden_dim)
        # Slot embeddings: we need up to ~80 for large molecules
        self.slot_emb = nn.Embedding(128, hidden_dim)
        self.route_mlp = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.sign_mlp = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, atom_features, z, batch):
        z = z.clamp(0, 19)
        type_emb = self.atom_type_emb(z)  # [N, H]
        device = atom_features.device
        B = int(batch.max().item()) + 1
        K_per_mol = molecular_valence_electrons(z, batch)
        K_max = int(K_per_mol.max().item())
        C = self.feature_dim
        H = type_emb.shape[-1]

        # Pad atom features and embeddings per molecule
        # Find max atoms per molecule
        mol_sizes = torch.bincount(batch, minlength=B)  # [B]
        N_max = int(mol_sizes.max().item())

        # Build padded tensors: [B, N_max, ...]
        af_pad = torch.zeros(B, N_max, C, device=device)
        te_pad = torch.zeros(B, N_max, H, device=device)
        z_pad = torch.zeros(B, N_max, dtype=torch.long, device=device)
        atom_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)

        for b in range(B):
            mask_b = batch == b
            n_b = mol_sizes[b].item()
            af_pad[b, :n_b] = atom_features[mask_b]
            te_pad[b, :n_b] = type_emb[mask_b]
            z_pad[b, :n_b] = z[mask_b]
            atom_mask[b, :n_b] = True

        # Build slot embeddings for the whole batch
        slot_emb = self.slot_emb(torch.arange(K_max, device=device))  # [K_max, H]

        # Compute coefficients per molecule
        O = torch.zeros(B, K_max, C, device=device)
        coeff_full = torch.zeros(B, K_max, N_max, device=device)
        mask = torch.zeros(B, K_max, dtype=torch.bool, device=device)
        coeff_list = []

        for b in range(B):
            Kb = int(K_per_mol[b])
            Nb = int(mol_sizes[b])
            mask[b, :Kb] = True
            if Kb == 0 or Nb == 0:
                coeff_list.append(torch.zeros(Kb, Nb, device=device))
                continue

            af = af_pad[b, :Nb]   # [Nb, C]
            te = te_pad[b, :Nb]   # [Nb, H]
            se = slot_emb[:Kb]     # [Kb, H]

            # Expand to [Nb, Kb, C+2H]
            af_exp = af.unsqueeze(1).expand(Nb, Kb, C)
            te_exp = te.unsqueeze(1).expand(Nb, Kb, H)
            se_exp = se.unsqueeze(0).expand(Nb, Kb, H)
            concat = torch.cat([af_exp, te_exp, se_exp], dim=-1)

            # Routing
            a = F.softmax(self.route_mlp(concat).squeeze(-1), dim=0)  # softmax over atoms
            s = torch.tanh(self.sign_mlp(concat).squeeze(-1))
            c = a * s  # [Nb, Kb]
            # Normalize: sum_i |c_{ki}| = 1 per slot
            norm = c.abs().sum(dim=0, keepdim=True).clamp(min=1e-8)
            c = c / norm

            coeff_full[b, :Kb, :Nb] = c.T  # [Kb, Nb]
            O[b, :Kb] = c.T @ af  # [Kb, Nb] @ [Nb, C] = [Kb, C]
            coeff_list.append(c.T)

        return {
            "O": O,
            "coeff": coeff_full,
            "mask": mask,
            "atom_mask": atom_mask,
            "K_per_mol": K_per_mol,
            "coeff_flat_list": coeff_list,
        }
