"""Baseline models for MTO-Net ablation comparisons.

Baselines:
  - Direct readout (no MTO)
  - Sum pooling
  - Attention pooling
  - Fixed-K latent token (no valence adaptivity)
  - MTO without sign
  - Full MTO (via main module)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectReadoutBaseline(nn.Module):
    """DetaNet backbone -> atom features -> global pool -> MLP -> properties."""

    def __init__(self, backbone, tasks: dict, hidden_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = backbone.num_features
        self.readout = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, dim) for name, dim in tasks.items()
        })

    def forward(self, z, pos, batch=None):
        out = self.backbone(z=z, pos=pos, batch=batch)
        af = out["atom_features"]  # [N, C]
        # Sum pool per molecule
        B = int(batch.max().item()) + 1 if batch is not None else 1
        pooled = torch.zeros(B, self.feature_dim, device=af.device)
        if batch is not None:
            pooled.scatter_add_(0, batch.unsqueeze(-1).expand(-1, self.feature_dim), af)
        else:
            pooled = af.sum(dim=0, keepdim=True)
        h = self.readout(pooled)
        return {name: head(h) for name, head in self.heads.items()}


class SumPoolingBaseline(DirectReadoutBaseline):
    """Alias: direct sum pooling readout."""
    pass


class AttentionPoolingBaseline(nn.Module):
    """DetaNet + learned attention pooling -> MLP -> properties."""

    def __init__(self, backbone, tasks: dict, hidden_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = backbone.num_features
        self.attn = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.readout = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, dim) for name, dim in tasks.items()
        })

    def forward(self, z, pos, batch=None):
        out = self.backbone(z=z, pos=pos, batch=batch)
        af = out["atom_features"]
        if batch is None:
            batch = torch.zeros(len(z), dtype=torch.long, device=af.device)
        B = int(batch.max().item()) + 1

        pooled = torch.zeros(B, self.feature_dim, device=af.device)
        for b in range(B):
            mask_b = (batch == b)
            af_b = af[mask_b]
            w = F.softmax(self.attn(af_b), dim=0)
            pooled[b] = (w * af_b).sum(dim=0)

        h = self.readout(pooled)
        return {name: head(h) for name, head in self.heads.items()}


class FixedKTokenBaseline(nn.Module):
    """Fixed-K latent tokens (no valence adaptivity) + DetaNet."""

    def __init__(self, backbone, tasks: dict, K_fixed: int = 20,
                 feature_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.backbone = backbone
        self.K = K_fixed
        self.feature_dim = feature_dim

        # Simple token routing
        self.slot_emb = nn.Embedding(K_fixed, hidden_dim)
        self.atom_type_emb = nn.Embedding(20, hidden_dim)
        self.route_mlp = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim),
            ) for name, dim in tasks.items()
        })

    def forward(self, z, pos, batch=None):
        out = self.backbone(z=z, pos=pos, batch=batch)
        af = out["atom_features"]
        if batch is None:
            batch = torch.zeros(len(z), dtype=torch.long, device=af.device)
        B = int(batch.max().item()) + 1
        device = af.device

        z_clamped = z.clamp(0, 19)
        type_emb = self.atom_type_emb(z_clamped)
        slot_emb = self.slot_emb(torch.arange(self.K, device=device))

        K = self.K
        O = torch.zeros(B, K, self.feature_dim, device=device)
        for b in range(B):
            mask = (batch == b)
            af_b = af[mask]
            te_b = type_emb[mask]
            n_a = af_b.shape[0]

            af_exp = af_b.unsqueeze(0).expand(K, n_a, -1)
            te_exp = te_b.unsqueeze(0).expand(K, n_a, -1)
            se_exp = slot_emb.unsqueeze(1).expand(K, n_a, -1)
            concat = torch.cat([af_exp, te_exp, se_exp], dim=-1)

            a_ki = F.softmax(self.route_mlp(concat).squeeze(-1), dim=-1)
            O[b] = a_ki @ af_b

        preds = {}
        for name, head in self.heads.items():
            preds[name] = head(O).sum(dim=1)
        return {**preds, "O": O}
