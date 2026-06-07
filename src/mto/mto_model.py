"""Full MTO-Net model: DetaNet backbone + ValenceAdaptiveMTO + readout heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compat import *  # noqa
from .detanet_adapter import DetaNetBackboneAdapter
from .mto_module import ValenceAdaptiveMTO
from .mto_readout import MultiHeadReadout


class MTONet(nn.Module):
    """Complete MTO-Net model.

    Pipeline:
      1. DetaNetBackboneAdapter: z, pos -> atom features S (invariant scalars)
      2. ValenceAdaptiveMTO: atom features -> MTO slot features O [B, K_max, C]
      3. MultiHeadReadout: O -> property predictions (mu, alpha, ir, raman, uv)
    """

    def __init__(
        self,
        feature_dim: int = 128,
        mto_hidden_dim: int = 64,
        readout_hidden_dim: int = 128,
        tasks: dict | None = None,
        detanet_kwargs: dict | None = None,
    ):
        super().__init__()
        dk = detanet_kwargs or {}
        self.backbone = DetaNetBackboneAdapter(num_features=feature_dim, **dk)
        self.mto = ValenceAdaptiveMTO(feature_dim=feature_dim, hidden_dim=mto_hidden_dim)
        self.tasks = tasks or {"mu": 3, "alpha": 6}
        self.readout = MultiHeadReadout(feature_dim, readout_hidden_dim, self.tasks)
        self.feature_dim = feature_dim

    def forward(self, z, pos, batch=None, edge_index=None, return_mto=False, mask_slot=None):
        backbone_out = self.backbone(z=z, pos=pos, batch=batch, edge_index=edge_index)
        atom_features = backbone_out["atom_features"]

        mto_out = self.mto(atom_features=atom_features, z=z, batch=backbone_out["batch"])
        O = mto_out["O"]
        mask = mto_out["mask"]

        # Slot intervention: zero out one slot
        if mask_slot is not None:
            O = O.clone()
            O[:, mask_slot, :] = 0.0

        preds = self.readout(O, mask)

        if return_mto:
            return {
                **preds,
                "O": O,
                "coeff": mto_out["coeff"],
                "mask": mask,
                "atom_mask": mto_out["atom_mask"],
                "K_per_mol": mto_out["K_per_mol"],
                "atom_tensors": backbone_out["atom_tensors"],
                "atom_features": atom_features,
            }
        return preds


class MTOWithoutSign(nn.Module):
    """MTO-Net variant without signed routing (ablation)."""

    def __init__(
        self,
        feature_dim: int = 128,
        mto_hidden_dim: int = 64,
        readout_hidden_dim: int = 128,
        tasks: dict | None = None,
        detanet_kwargs: dict | None = None,
    ):
        super().__init__()
        dk = detanet_kwargs or {}
        self.backbone = DetaNetBackboneAdapter(num_features=feature_dim, **dk)
        self.tasks = tasks or {"mu": 3, "alpha": 6}
        self.readout = MultiHeadReadout(feature_dim, readout_hidden_dim, self.tasks)
        self.feature_dim = feature_dim

        # Simplified MTO: no sign, just softmax attention
        num_types = 20
        self.atom_type_emb = nn.Embedding(num_types, mto_hidden_dim)
        self.route_mlp = nn.Sequential(
            nn.Linear(feature_dim + mto_hidden_dim + mto_hidden_dim, mto_hidden_dim),
            nn.SiLU(),
            nn.Linear(mto_hidden_dim, 1),
        )

    def forward(self, z, pos, batch=None, edge_index=None, return_mto=False, mask_slot=None):
        backbone_out = self.backbone(z=z, pos=pos, batch=batch, edge_index=edge_index)
        atom_features = backbone_out["atom_features"]
        b = backbone_out["batch"]

        # Simplified routing (no sign, no valence adaptivity)
        from .valence import molecular_valence_electrons
        K_per_mol = molecular_valence_electrons(z, b)
        K_max = int(K_per_mol.max().item())
        B = int(b.max().item()) + 1
        device = atom_features.device

        z_clamped = z.clamp(0, 19)
        type_emb = self.atom_type_emb(z_clamped)
        slot_ids = torch.arange(K_max, device=device)
        slot_emb = self.atom_type_emb(slot_ids.clamp(0, 19))

        mask = torch.zeros(B, K_max, dtype=torch.bool, device=device)
        for mb in range(B):
            mask[mb, :K_per_mol[mb]] = True

        O = torch.zeros(B, K_max, self.feature_dim, device=device)
        coeff = torch.zeros(B, K_max, max((b == 0).sum().item(), 1), device=device)
        atom_mask = torch.zeros_like(coeff[:, 0, :])

        for mb in range(B):
            mol_mask = (b == mb)
            af = atom_features[mol_mask]
            te = type_emb[mol_mask]
            Kb = int(K_per_mol[mb])
            n_a = af.shape[0]

            af_exp = af.unsqueeze(0).expand(Kb, n_a, -1)
            te_exp = te.unsqueeze(0).expand(Kb, n_a, -1)
            se_exp = slot_emb[:Kb].unsqueeze(1).expand(Kb, n_a, -1)
            concat = torch.cat([af_exp, te_exp, se_exp], dim=-1)
            a_ki = F.softmax(self.route_mlp(concat).squeeze(-1), dim=-1)
            norm = a_ki.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
            a_ki = a_ki / norm
            O[mb, :Kb] = a_ki @ af
            coeff[mb, :Kb, :n_a] = a_ki

        if mask_slot is not None:
            O = O.clone()
            O[:, mask_slot, :] = 0.0

        preds = self.readout(O, mask)

        if return_mto:
            return {
                **preds,
                "O": O, "coeff": coeff, "mask": mask,
                "atom_mask": atom_mask, "K_per_mol": K_per_mol,
                "atom_tensors": backbone_out["atom_tensors"],
                "atom_features": atom_features,
            }
        return preds
