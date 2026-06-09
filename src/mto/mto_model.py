"""Full MTO-Net model: DetaNet backbone + ValenceAdaptiveMTO + readout heads.

Supports configurable activity gate modes: none, simple, fermi_dirac.
Supports Stage A (mu, alpha), Stage B (+IR, +Raman), Stage C (+UV).
Spectral tasks use the same readout architecture with appropriate output dimensions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .compat import *
from .detanet_adapter import DetaNetBackboneAdapter
from .mto_module import ValenceAdaptiveMTO
from .mto_readout import MultiHeadReadout


class MTONet(nn.Module):
    def __init__(self, feature_dim=128, mto_hidden_dim=64, readout_hidden_dim=128,
                 tasks=None, detanet_kwargs=None,
                 use_activity_gate=True, activity_mode="simple"):
        super().__init__()
        dk = detanet_kwargs or {}
        self.backbone = DetaNetBackboneAdapter(num_features=feature_dim, **dk)
        self.mto = ValenceAdaptiveMTO(
            feature_dim=feature_dim,
            hidden_dim=mto_hidden_dim,
            use_activity_gate=use_activity_gate,
            activity_mode=activity_mode,
        )
        self.tasks = tasks or {"mu": 3, "alpha": 9}
        self.readout = MultiHeadReadout(feature_dim, readout_hidden_dim, self.tasks)
        self.feature_dim = feature_dim
        self.use_activity_gate = use_activity_gate
        self.activity_mode = activity_mode

    def forward(self, z, pos, batch=None, edge_index=None,
                return_mto=False, mask_slot=None, theta=0.5):
        bb = self.backbone(z=z, pos=pos, batch=batch, edge_index=edge_index)
        af = bb["atom_features"]
        mto_out = self.mto(atom_features=af, z=z, batch=bb["batch"], theta=theta)

        O = mto_out["O"]
        m = mto_out["mask"]

        if mask_slot is not None:
            O = O.clone()
            O[:, mask_slot, :] = 0.0

        preds = self.readout(O, m)

        if return_mto:
            return {
                **preds,
                "O": O,
                "O_raw": mto_out.get("O_raw", O),
                "coeff": mto_out["coeff"],
                "mask": m,
                "atom_mask": mto_out["atom_mask"],
                "K_per_mol": mto_out["K_per_mol"],
                "activity": mto_out["activity"],
                "epsilon": mto_out.get("epsilon"),
                "mu_chem": mto_out.get("mu"),
                "routing_logits": mto_out.get("routing_logits", []),
                "atom_tensors": bb["atom_tensors"],
                "atom_features": af,
            }
        return preds
