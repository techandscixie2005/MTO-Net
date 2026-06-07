import torch
import torch.nn as nn
import torch.nn.functional as F

from .valence import molecular_valence_electrons


class ValenceAdaptiveMTO(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        num_types = 20  # 0..19 covers H(1)..K(19)
        self.atom_type_emb = nn.Embedding(num_types, hidden_dim)
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
        # batch: [N_atoms]
        B = int(batch.max().item()) + 1
        K_per_mol = molecular_valence_electrons(z, batch)
        K_max = int(K_per_mol.max().item())
        device = atom_features.device

        z_clamped = z.clamp(0, 19)
        type_emb = self.atom_type_emb(z_clamped)  # [N, hidden_dim]

        # Build slot embeddings [K_max, hidden_dim]
        slot_ids = torch.arange(K_max, device=device)
        slot_emb = self.atom_type_emb(slot_ids.clamp(0, 19))  # simple: reuse type emb

        # Per-molecule padding mask for slots
        mask = torch.zeros(B, K_max, dtype=torch.bool, device=device)
        for b in range(B):
            mask[b, : K_per_mol[b]] = True

        # Per-molecule atom mask
        n_atoms = torch.zeros(B, dtype=torch.long, device=device)
        n_atoms.scatter_add_(0, batch, torch.ones_like(batch))
        N_max = int(n_atoms.max().item())

        atom_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)
        atom_idx_in_mol = torch.zeros_like(batch)
        mol_counts = torch.zeros(B, dtype=torch.long, device=device)
        for i in range(len(batch)):
            mol = int(batch[i])
            atom_idx_in_mol[i] = mol_counts[mol]
            mol_counts[mol] += 1
            atom_mask[mol, atom_idx_in_mol[i]] = True

        # Compute coefficients c_{k,i} for each (slot k, atom i) pair
        coeff_raw = torch.zeros(B, K_max, N_max, device=device)  # padded
        coeff_flat_list = []

        for b in range(B):
            mol_atoms = (batch == b)
            n_a = int(n_atoms[b])
            K_b = int(K_per_mol[b])
            af = atom_features[mol_atoms]  # [n_a, C]
            te = type_emb[mol_atoms]       # [n_a, H]

            # Expand to [K_b, n_a, ...]
            af_exp = af.unsqueeze(0).expand(K_b, n_a, -1)    # [K_b, n_a, C]
            te_exp = te.unsqueeze(0).expand(K_b, n_a, -1)    # [K_b, n_a, H]
            se_exp = slot_emb[:K_b].unsqueeze(1).expand(K_b, n_a, -1)  # [K_b, n_a, H]

            concat = torch.cat([af_exp, te_exp, se_exp], dim=-1)  # [K_b, n_a, C+2H]

            a_ki = F.softmax(self.route_mlp(concat).squeeze(-1), dim=-1)  # [K_b, n_a]
            s_ki = torch.tanh(self.sign_mlp(concat).squeeze(-1))           # [K_b, n_a]
            c_ki = a_ki * s_ki  # [K_b, n_a]

            # Normalize: sum_i |c_ki| = 1 per slot
            norm = c_ki.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
            c_ki = c_ki / norm

            coeff_flat_list.append(c_ki)
            coeff_raw[b, :K_b, :n_a] = c_ki

        # Build O_k = sum_i c_{k,i} * atom_feature_i
        O = torch.zeros(B, K_max, self.feature_dim, device=device)

        for b in range(B):
            mol_atoms = (batch == b)
            n_a = int(n_atoms[b])
            K_b = int(K_per_mol[b])
            af_padded = torch.zeros(N_max, self.feature_dim, device=device)
            af_padded[:n_a] = atom_features[mol_atoms]
            c = coeff_raw[b, :K_b, :n_a]
            O[b, :K_b] = c @ af_padded[:n_a]

        return {
            "O": O,
            "coeff": coeff_raw,
            "mask": mask,
            "atom_mask": atom_mask,
            "K_per_mol": K_per_mol,
            "coeff_flat_list": coeff_flat_list,
        }
