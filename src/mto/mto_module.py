"""Valence-adaptive MTO module with vectorized routing."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .valence import molecular_valence_electrons


class ValenceAdaptiveMTO(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        num_types = 20
        self.atom_type_emb = nn.Embedding(num_types, hidden_dim)
        self.slot_emb = nn.Embedding(64, hidden_dim)
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
        # atom_features: [N_atoms, C]
        # z: [N_atoms]
        # batch: [N_atoms] —  molecule index for each atom
        B = int(batch.max().item()) + 1
        K_per_mol = molecular_valence_electrons(z, batch)  # [B]
        K_max = int(K_per_mol.max().item())
        device = atom_features.device

        z_clamped = z.clamp(0, 19)
        type_emb = self.atom_type_emb(z_clamped)  # [N, H]
        slot_emb = self.slot_emb(torch.arange(K_max, device=device))  # [K_max, H]

        # For each molecule, compute per-atom contributions via scatter
        # Build output O and coeff matrices using segmented operations
        N_atoms = atom_features.shape[0]
        C = self.feature_dim
        H = type_emb.shape[-1]

        # Compute atom-slot interactions for all atoms simultaneously
        # For each atom i, compute its interaction with slots k=0..K_max-1
        # Expand: [N, K_max, C+2H] where each row is [af_i, type_emb_i, slot_emb_k]
        af_exp = atom_features.unsqueeze(1).expand(-1, K_max, -1)  # [N, K_max, C]
        te_exp = type_emb.unsqueeze(1).expand(-1, K_max, -1)     # [N, K_max, H]
        se_exp = slot_emb.unsqueeze(0).expand(N_atoms, -1, -1)   # [N, K_max, H]
        concat = torch.cat([af_exp, te_exp, se_exp], dim=-1)     # [N, K_max, C+2H]

        # Softmax routing over atoms per molecule per slot
        route_raw = self.route_mlp(concat).squeeze(-1)  # [N, K_max]
        sign_raw = torch.tanh(self.sign_mlp(concat).squeeze(-1))  # [N, K_max]

        # Build mask: which (atom, slot) pairs are valid
        atom_has_slot = torch.zeros(B, K_max, dtype=torch.bool, device=device)
        for b in range(B):
            atom_has_slot[b, :K_per_mol[b]] = True

        # Per-atom slot validity
        slot_valid = torch.zeros(N_atoms, K_max, dtype=torch.bool, device=device)
        for b in range(B):
            mask_b = (batch == b)
            slot_valid[mask_b, :K_per_mol[b]] = True

        # Apply mask before softmax
        route_masked = route_raw.masked_fill(~slot_valid, float('-inf'))

        # Softmax over atoms per slot — need to do this within each molecule
        # Use scatter-softmax pattern
        # For slot k, softmax over atoms belonging to molecule b
        coeff = torch.zeros(N_atoms, K_max, device=device)

        for b in range(B):
            mask_b = (batch == b)
            n_b = mask_b.sum().item()
            K_b = int(K_per_mol[b])
            route_b = route_masked[mask_b, :K_b]  # [n_b, K_b]
            sign_b = sign_raw[mask_b, :K_b]         # [n_b, K_b]
            a_kib = F.softmax(route_b, dim=0)  # softmax over atoms
            c_kib = a_kib * sign_b
            # Normalize
            norm = c_kib.abs().sum(dim=0, keepdim=True).clamp(min=1e-8)
            c_kib = c_kib / norm
            coeff[mask_b, :K_b] = c_kib

        # Build O[k] = sum_i c_{k,i} * af_i
        O = torch.zeros(B, K_max, C, device=device)
        for b in range(B):
            mask_b = (batch == b)
            K_b = int(K_per_mol[b])
            af_b = atom_features[mask_b]   # [n_b, C]
            c_b = coeff[mask_b, :K_b]       # [n_b, K_b]
            O[b, :K_b] = c_b.T @ af_b       # [K_b, n_b] @ [n_b, C] = [K_b, C]

        mask = torch.zeros(B, K_max, dtype=torch.bool, device=device)
        for b in range(B):
            mask[b, :K_per_mol[b]] = True

        # Build padded coeff matrix
        n_max = int(torch.bincount(batch).max().item())
        coeff_padded = torch.zeros(B, K_max, n_max, device=device)
        atom_mask_padded = torch.zeros(B, n_max, dtype=torch.bool, device=device)
        for b in range(B):
            mask_b = (batch == b)
            n_b = mask_b.sum().item()
            atom_idx_b = torch.where(mask_b)[0]
            coeff_padded[b, :int(K_per_mol[b]), :n_b] = coeff[atom_idx_b, :int(K_per_mol[b])].T
            atom_mask_padded[b, :n_b] = True

        # coeff_flat_list for testing
        coeff_list = []
        for b in range(B):
            mask_b = (batch == b)
            coeff_list.append(coeff[mask_b, :int(K_per_mol[b])].T)

        return {
            "O": O,
            "coeff": coeff_padded,
            "mask": mask,
            "atom_mask": atom_mask_padded,
            "K_per_mol": K_per_mol,
            "coeff_flat_list": coeff_list,
        }
