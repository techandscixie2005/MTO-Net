"""Full MTO-Net model: DetaNet backbone + ValenceAdaptiveMTO + readout heads."""
import torch, torch.nn as nn, torch.nn.functional as F
from .compat import *
from .detanet_adapter import DetaNetBackboneAdapter
from .mto_module import ValenceAdaptiveMTO
from .mto_readout import MultiHeadReadout

class MTONet(nn.Module):
    def __init__(self, feature_dim=128, mto_hidden_dim=64, readout_hidden_dim=128,
                 tasks=None, detanet_kwargs=None):
        super().__init__()
        dk = detanet_kwargs or {}
        self.backbone = DetaNetBackboneAdapter(num_features=feature_dim, **dk)
        self.mto = ValenceAdaptiveMTO(feature_dim=feature_dim, hidden_dim=mto_hidden_dim)
        self.tasks = tasks or {"mu": 3, "alpha": 9}
        self.readout = MultiHeadReadout(feature_dim, readout_hidden_dim, self.tasks)
        self.feature_dim = feature_dim

    def forward(self, z, pos, batch=None, edge_index=None, return_mto=False, mask_slot=None):
        bb = self.backbone(z=z, pos=pos, batch=batch, edge_index=edge_index)
        af = bb["atom_features"]
        mto_out = self.mto(atom_features=af, z=z, batch=bb["batch"])
        O = mto_out["O"]
        m = mto_out["mask"]
        if mask_slot is not None:
            O = O.clone()
            O[:, mask_slot, :] = 0.0
        preds = self.readout(O, m)
        if return_mto:
            return {**preds, "O": O, "coeff": mto_out["coeff"], "mask": m,
                    "atom_mask": mto_out["atom_mask"], "K_per_mol": mto_out["K_per_mol"],
                    "atom_tensors": bb["atom_tensors"], "atom_features": af}
        return preds
