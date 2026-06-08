"""Spectrum builder: generates IR/Raman/UV from physical quantities when available.

QM9S has only dipole and polarizability labels.
Spectrum builder is a stub for Stage B/C when IR/Raman/UV labels are absent.
Direct spectrum regression is marked as fallback.
"""
import torch
import torch.nn as nn


class SpectrumBuilder(nn.Module):
    """Build molecular spectra from MTO representations.

    When physical labels (Hessian, normal modes, transition dipoles) are
    available, uses the MTO -> physical quantities -> spectra pipeline.
    When labels are absent, falls back to direct MTO -> spectrum regression.

    Currently: only direct regression mode is active because QM9S has no
    IR/Raman/UV physical labels (confirmed in Phase 0 dataset audit).
    """

    def __init__(self, feature_dim, num_bins=3501, mode="regression"):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_bins = num_bins
        self.mode = mode  # "regression" (current) or "physical" (future)

        self.regression_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, num_bins),
        )

    def forward(self, O, mask):
        """Build spectrum from MTO features.

        Args:
            O: [B, K_max, C] MTO features
            mask: [B, K_max] valid slot mask

        Returns:
            spectrum: [B, num_bins] predicted spectral intensities
        """
        if self.mode == "regression":
            return self._regression(O, mask)
        else:
            return self._physical_pipeline(O, mask)

    def _regression(self, O, mask):
        # Pool MTO features across slots
        O_masked = O * mask.unsqueeze(-1).float()
        pooled = O_masked.sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return self.regression_net(pooled)

    def _physical_pipeline(self, O, mask):
        raise NotImplementedError(
            "Physical pipeline requires Hessian/normal mode labels "
            "which are not available in QM9S."
        )

    @property
    def is_fallback(self):
        """True when using direct regression fallback."""
        return self.mode == "regression"
