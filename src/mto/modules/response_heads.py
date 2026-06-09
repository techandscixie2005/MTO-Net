"""Update to spectrum_builder supporting IR/Raman/UV spectral output from MTO features.

Spectral tasks are treated as response-level supervision from the QM9S dataset.
Physical derivative-based construction is not attempted without explicit labels.
"""
import torch
import torch.nn as nn


class SpectrumHead(nn.Module):
    """Single spectrum prediction head from pooled MTO features."""

    def __init__(self, feature_dim, num_bins, hidden_dim=256, name=""):
        super().__init__()
        self.name = name
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, O, mask=None):
        """O: [B, K_max, C], mask: [B, K_max]"""
        if mask is not None:
            O = O * mask.unsqueeze(-1).float()
        # Pool over MTO slots
        pooled = O.sum(dim=1)  # [B, C]
        return self.net(pooled)


class MultiSpectrumBuilder(nn.Module):
    """Build IR, Raman, UV spectra from MTO features.

    Each spectrum type has its own head with appropriate bin count.
    This is response-level spectral supervision (not physical derivative-based).
    """

    def __init__(self, feature_dim, tasks=None, hidden_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.tasks = tasks or {}
        self.heads = nn.ModuleDict()

        # Default bin counts from QM9S Figshare dataset
        bin_counts = {
            "ir": 3501,
            "raman": 3501,
            "uv": 701,
        }

        for name in self.tasks:
            if name in ("mu", "alpha"):
                continue
            num_bins = bin_counts.get(name, 3501)
            self.heads[name] = SpectrumHead(
                feature_dim, num_bins, hidden_dim=hidden_dim, name=name
            )

    def forward(self, O, mask=None):
        """Returns dict of task_name -> [B, num_bins] tensor."""
        results = {}
        for name, head in self.heads.items():
            results[name] = head(O, mask)
        return results


# Keep backward compatibility alias
SpectrumBuilder = MultiSpectrumBuilder
