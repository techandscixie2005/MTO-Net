"""Adapter to expose DetaNet atom-level hidden representations for MTO."""
import os
import sys
import torch
import torch.nn as nn

# Resolve through symlinks to find the real third_party path
_real_file = os.path.realpath(__file__)
_project_root = os.path.abspath(os.path.join(os.path.dirname(_real_file), "..", ".."))
_detanet_path = os.path.join(_project_root, "third_party", "DetaNet")
if _detanet_path not in sys.path:
    sys.path.insert(0, _detanet_path)

# Apply compat patches before importing DetaNet
from . import compat as _compat  # noqa: F401, E402

from detanet_model.detanet import DetaNet  # noqa: E402


class DetaNetBackboneAdapter(nn.Module):
    def __init__(self, num_features=128, maxl=3, num_block=3, rc=5.0,
                 max_atomic_number=9, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.backbone = DetaNet(
            num_features=num_features, maxl=maxl, num_block=num_block,
            rc=rc, max_atomic_number=max_atomic_number,
            out_type="latent", summation=False, scalar_outsize=0,
            irreps_out=None, scale=None, device=device)
        self.num_features = num_features
        self.vdim = self.backbone.vdim

    def forward(self, z, pos, batch=None, edge_index=None):
        S, T = self.backbone(z=z, pos=pos, batch=batch, edge_index=edge_index)
        return {
            "atom_tensors": {"h0": S, "T": T},
            "atom_features": S,
            "batch": batch if batch is not None else torch.zeros(len(z), dtype=torch.long, device=z.device),
            "z": z,
            "pos": pos,
        }
