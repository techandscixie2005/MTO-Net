"""MTO readout heads for Stage A (mu+alpha), Stage B (+IR+Raman), Stage C (+UV)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PropertyReadout(nn.Module):
    """Single-property readout from MTO slot features.

    Input: O [B, K_max, feature_dim]
    Output: property prediction (shape depends on property)
    """

    def __init__(self, feature_dim: int, hidden_dim: int, out_dim: int, name: str = ""):
        super().__init__()
        self.name = name
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, O, mask=None):
        # O: [B, K_max, C]
        # mask: [B, K_max]
        x = self.net(O)  # [B, K_max, out_dim]
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        # Pool across slots: sum contributions
        out = x.sum(dim=1)  # [B, out_dim]
        return out


class MultiHeadReadout(nn.Module):
    """Multi-property readout for MTO-Net.

    Uses one PropertyReadout per task, all reading from the same
    MTO slot features O [B, K_max, feature_dim].
    """

    def __init__(self, feature_dim: int, hidden_dim: int, tasks: dict):
        """
        Args:
            feature_dim: MTO slot feature dimension.
            hidden_dim: hidden dim for each readout head.
            tasks: dict mapping task_name -> out_dim.
        """
        super().__init__()
        self.tasks = tasks
        self.heads = nn.ModuleDict()
        for name, out_dim in tasks.items():
            self.heads[name] = PropertyReadout(feature_dim, hidden_dim, out_dim, name=name)

    def forward(self, O, mask=None):
        return {name: head(O, mask) for name, head in self.heads.items()}


# Standard task dimensions
STAGE_A_TASKS = {
    "mu": 3,      # dipole vector
    "alpha": 6,   # polarizability (upper triangular)
}

STAGE_B_TASKS = {
    "mu": 3,
    "alpha": 6,
    "ir": 3501,   # IR spectrum (500-4000 cm^-1, 1 cm^-1 steps)
    "raman": 3501,
}

STAGE_C_TASKS = {
    "mu": 3,
    "alpha": 6,
    "ir": 3501,
    "raman": 3501,
    "uv": 601,    # UV spectrum (1.5-13.5 eV, 0.02 eV steps)
}


def make_readout(stage: str, feature_dim: int, hidden_dim: int = 128,
                 ir_bins: int = 3501, raman_bins: int = 3501,
                 uv_bins: int = 601) -> MultiHeadReadout:
    """Factory for stage-specific readout head."""
    if stage == "stage_a":
        tasks = STAGE_A_TASKS
    elif stage == "stage_b":
        tasks = STAGE_B_TASKS
    elif stage == "stage_c":
        tasks = STAGE_C_TASKS
    else:
        raise ValueError(f"Unknown stage: {stage}")

    if "ir" in tasks:
        tasks["ir"] = ir_bins
    if "raman" in tasks:
        tasks["raman"] = raman_bins
    if "uv" in tasks:
        tasks["uv"] = uv_bins

    return MultiHeadReadout(feature_dim, hidden_dim, tasks)
