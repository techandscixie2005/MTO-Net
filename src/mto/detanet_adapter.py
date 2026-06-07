"""Adapter to expose DetaNet atom-level hidden representations for MTO."""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "DetaNet"))

from detanet_model.detanet import DetaNet


class DetaNetBackboneAdapter(nn.Module):
    def __init__(
        self,
        num_features: int = 128,
        maxl: int = 3,
        num_block: int = 3,
        rc: float = 5.0,
        max_atomic_number: int = 9,
        device: torch.device = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = DetaNet(
            num_features=num_features,
            maxl=maxl,
            num_block=num_block,
            rc=rc,
            max_atomic_number=max_atomic_number,
            out_type="latent",
            summation=False,
            scalar_outsize=0,
            irreps_out=None,
            device=device,
        )
        self.num_features = num_features

    def forward(self, z, pos, batch=None, edge_index=None):
        S, T = self.backbone(z=z, pos=pos, batch=batch, edge_index=edge_index)
        return {
            "atom_features": S,
            "T": T,
            "batch": batch if batch is not None else torch.zeros(len(z), dtype=torch.long, device=z.device),
            "z": z,
            "pos": pos,
        }
