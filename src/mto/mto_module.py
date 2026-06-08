"""Valence-adaptive MTO module with signed assembly, gated nonlinearity, activity gate.

Per TOTAL.md 8: center-free, l-specific signed, normalized assembly.
Per TOTAL.md 9: configurable activity gate (none/simple/fermi_dirac).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .valence import molecular_valence_electrons
from .activity_gate import ActivityGate


class ValenceAdaptiveMTO(nn.Module):
    """Valence-adaptive MTO: K = total valence electrons per molecule."""

    def __init__(self, feature_dim, hidden_dim=64,
                 use_activity_gate=True, activity_mode="simple"):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_activity_gate = use_activity_gate
        self.activity_mode = activity_mode
        num_types = 20
        self.atom_type_emb = nn.Embedding(num_types, hidden_dim)
        self.slot_emb = nn.Embedding(128, hidden_dim)

        # Shared invariant routing MLP
        self.route_mlp = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # l-specific sign MLPs
        self.sign_mlp_l0 = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.sign_mlp_l1 = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.sign_mlp_l2 = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Gated nonlinear relaxation (TOTAL 8.4)
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.nonlin_mlp = nn.Sequential(
            nn.Linear(feature_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Activity gate
        if use_activity_gate:
            self.activity_gate = ActivityGate(
                feature_dim, hidden_dim, mode=activity_mode)
        else:
            self.activity_gate = ActivityGate(
                feature_dim, hidden_dim, mode="none")

        self.alpha_nonlin = 0.01  # small init, near-linear start

    def forward(self, atom_features, z, batch, theta=0.5):
        z = z.clamp(0, 19)
        type_emb = self.atom_type_emb(z)
        device = atom_features.device
        B = int(batch.max().item()) + 1
        K_per_mol = molecular_valence_electrons(z, batch)
        K_max = int(K_per_mol.max().item())
        C = self.feature_dim
        H = type_emb.shape[-1]

        mol_sizes = torch.bincount(batch, minlength=B)
        N_max = int(mol_sizes.max().item())

        af_pad = torch.zeros(B, N_max, C, device=device)
        te_pad = torch.zeros(B, N_max, H, device=device)
        atom_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)
        for b in range(B):
            mask_b = batch == b
            n_b = mol_sizes[b].item()
            af_pad[b, :n_b] = atom_features[mask_b]
            te_pad[b, :n_b] = type_emb[mask_b]
            atom_mask[b, :n_b] = True

        slot_emb = self.slot_emb(torch.arange(K_max, device=device))

        O = torch.zeros(B, K_max, C, device=device)
        coeff_full = torch.zeros(B, K_max, N_max, device=device)
        mask = torch.zeros(B, K_max, dtype=torch.bool, device=device)
        coeff_list = []
        routing_logits_list = []

        for b in range(B):
            Kb = int(K_per_mol[b])
            Nb = int(mol_sizes[b])
            mask[b, :Kb] = True
            if Kb == 0 or Nb == 0:
                coeff_list.append(torch.zeros(Kb, Nb, device=device))
                routing_logits_list.append(torch.zeros(Nb, Kb, device=device))
                continue

            af = af_pad[b, :Nb]
            te = te_pad[b, :Nb]
            se = slot_emb[:Kb]

            af_exp = af.unsqueeze(1).expand(Nb, Kb, C)
            te_exp = te.unsqueeze(1).expand(Nb, Kb, H)
            se_exp = se.unsqueeze(0).expand(Nb, Kb, H)
            concat = torch.cat([af_exp, te_exp, se_exp], dim=-1)

            # Shared invariant routing
            e = self.route_mlp(concat).squeeze(-1)
            a = F.softmax(e, dim=0)

            # l-specific signs
            s0 = torch.tanh(self.sign_mlp_l0(concat).squeeze(-1))
            s1 = torch.tanh(self.sign_mlp_l1(concat).squeeze(-1))
            s2 = torch.tanh(self.sign_mlp_l2(concat).squeeze(-1))
            s_avg = (s0 + s1 + s2) / 3.0
            c_raw = a * s_avg

            norm = c_raw.abs().sum(dim=0, keepdim=True).clamp(min=1e-8)
            c = c_raw / norm

            routing_logits_list.append(e.T)
            coeff_full[b, :Kb, :Nb] = c.T
            O_linear = c.T @ af

            # Gated nonlinear relaxation
            P_norm = O_linear.norm(dim=-1)
            I_k = torch.cat([
                O_linear,
                P_norm.unsqueeze(-1),
                P_norm.unsqueeze(-1),
                torch.zeros(Kb, 1, device=device),
            ], dim=-1)

            gate = torch.sigmoid(self.gate_mlp(I_k))
            nonlin = self.nonlin_mlp(I_k)
            O_relaxed = O_linear + self.alpha_nonlin * gate * nonlin

            O[b, :Kb] = O_relaxed
            coeff_list.append(c.T)

        # Activity gate
        gate_kwargs = {}
        if self.activity_mode == "fermi_dirac":
            gate_kwargs["N_val"] = K_per_mol.float()
        n_k, O_active, epsilon, mu = self.activity_gate(
            O, mask, theta=theta, **gate_kwargs)

        return {
            "O": O_active,
            "O_raw": O,
            "coeff": coeff_full,
            "mask": mask,
            "atom_mask": atom_mask,
            "K_per_mol": K_per_mol,
            "coeff_flat_list": coeff_list,
            "routing_logits": routing_logits_list,
            "activity": n_k,
            "epsilon": epsilon,
            "mu": mu,
        }
