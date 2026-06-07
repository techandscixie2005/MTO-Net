"""Synthetic molecule builders for smoke testing and visualization."""

import torch


def make_ethanol():
    """Build ethanol (C2H6O) with approximate geometry."""
    z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1])
    pos = torch.tensor([
        [-0.750, 0.000, 0.000],
        [0.750, 0.000, 0.000],
        [1.450, 1.390, 0.000],
        [-1.200, 0.940, 0.000],
        [-1.200, -0.500, -0.880],
        [-1.200, -0.500, 0.880],
        [1.200, -0.500, -0.880],
        [1.200, -0.500, 0.880],
        [2.440, 1.240, 0.000],
    ], dtype=torch.float32)
    batch = torch.zeros(9, dtype=torch.long)
    return z, pos, batch


def make_formaldehyde():
    """Build formaldehyde (CH2O) with approximate geometry."""
    z = torch.tensor([6, 8, 1, 1])
    pos = torch.tensor([
        [0.000, 0.000, 0.000],
        [1.200, 0.000, 0.000],
        [-0.580, 0.950, 0.000],
        [-0.580, -0.950, 0.000],
    ], dtype=torch.float32)
    batch = torch.zeros(4, dtype=torch.long)
    return z, pos, batch


def make_water():
    """Build water (H2O) with approximate geometry."""
    z = torch.tensor([8, 1, 1])
    pos = torch.tensor([
        [0.000, 0.000, 0.117],
        [0.000, 0.757, -0.469],
        [0.000, -0.757, -0.469],
    ], dtype=torch.float32)
    batch = torch.zeros(3, dtype=torch.long)
    return z, pos, batch
