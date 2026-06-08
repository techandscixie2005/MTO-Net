"""MTO readout heads for Stage A (mu+alpha), Stage B (+IR+Raman), Stage C (+UV)."""
import torch, torch.nn as nn, torch.nn.functional as F

class PropertyReadout(nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim, name=""):
        super().__init__()
        self.name = name
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, O, mask=None):
        x = self.net(O)
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        return x.sum(dim=1)

class MultiHeadReadout(nn.Module):
    def __init__(self, feature_dim, hidden_dim, tasks):
        super().__init__()
        self.tasks = tasks
        self.heads = nn.ModuleDict()
        for name, out_dim in tasks.items():
            self.heads[name] = PropertyReadout(feature_dim, hidden_dim, out_dim, name=name)
    def forward(self, O, mask=None):
        return {name: head(O, mask) for name, head in self.heads.items()}

def make_readout(stage, feature_dim, hidden_dim=128, ir_bins=3501, raman_bins=3501, uv_bins=601):
    tasks = {
        "stage_a": {"mu": 3, "alpha": 9},
        "stage_b": {"mu": 3, "alpha": 9, "ir": ir_bins, "raman": raman_bins},
        "stage_c": {"mu": 3, "alpha": 9, "ir": ir_bins, "raman": raman_bins, "uv": uv_bins},
    }
    return MultiHeadReadout(feature_dim, hidden_dim, tasks[stage])
