"""QM9S spectral data loading: IR, Raman, UV from boraden CSV files.

Supports chunked reading, subset loading, and alignment to qm9s.pt molecules.
Uses real filenames (ir_boraden.csv, raman_boraden.csv, uv_boraden.csv).
"""
import csv
import os
import torch
import numpy as np


SPECTRAL_FILES = {
    "ir": "ir_boraden.csv",
    "raman": "raman_boraden.csv",
    "uv": "uv_boraden.csv",
}

SPECTRAL_GRID = {
    "ir": {"bins": 3501, "start": 0.0, "end": 3500.0, "step": 1.0},
    "raman": {"bins": 3501, "start": 0.0, "end": 3500.0, "step": 1.0},
    "uv": {"bins": 701, "start": 1.0, "end": 15.0, "step": 0.02},
}


def check_spectral_file(data_dir, task_name):
    """Check that a spectral CSV file exists. Returns full path or raises FileNotFoundError."""
    fname = SPECTRAL_FILES.get(task_name)
    if fname is None:
        raise ValueError(f"Unknown spectral task: {task_name}")
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Spectral file not found: {path}\n"
            f"Expected file for task '{task_name}'. "
            f"The QM9S Figshare dataset uses the spelling 'boraden' (not 'broaden')."
        )
    return path


def count_spectral_rows(data_dir, task_name):
    """Count data rows in a spectral CSV without loading all content."""
    path = check_spectral_file(data_dir, task_name)
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return max(0, count - 1)  # exclude header


def read_spectral_header(data_dir, task_name):
    """Read the header row of a spectral CSV."""
    path = check_spectral_file(data_dir, task_name)
    with open(path, "r") as f:
        reader = csv.reader(f)
        return next(reader)


def load_spectral_tensor(data_dir, task_name, max_rows=None, row_indices=None,
                         dtype=torch.float32):
    """Load spectral intensities from a CSV file into a tensor.

    Args:
        data_dir: path to the qm9s data directory
        task_name: "ir", "raman", or "uv"
        max_rows: if set, load only the first N rows
        row_indices: if set, load only specific row indices (0-based, after header)
        dtype: output tensor dtype

    Returns:
        spectral_tensor: [N, bins] float tensor
        grid: [bins] float tensor of frequency/wavelength values
    """
    path = check_spectral_file(data_dir, task_name)

    if row_indices is not None:
        row_indices = set(row_indices)

    intensities = []
    grid = None
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Parse grid from header (skip first mol_id column)
        grid = torch.tensor([float(x) for x in header[1:]], dtype=dtype)

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            if row_indices is not None and i not in row_indices:
                continue
            # Skip mol_id column (first), parse intensities
            try:
                vals = [float(x) for x in row[1:]]
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Failed to parse row {i} in {path}: {e}\n"
                    f"Row has {len(row)} columns, expected {len(header)}"
                )
            intensities.append(vals)

    if not intensities:
        raise ValueError(f"No data loaded from {path}")

    return torch.tensor(intensities, dtype=dtype), grid


def load_spectral_targets(data_dir, tasks, max_molecules=None,
                          dtype=torch.float32):
    """Load spectral targets for a list of task names.

    Args:
        data_dir: path to qm9s data directory
        tasks: list of task names, e.g. ["ir", "raman", "uv"]
        max_molecules: if set, load only first N rows
        dtype: output dtype

    Returns:
        targets: dict mapping task_name -> [N, bins] tensor
        grids: dict mapping task_name -> [bins] tensor
    """
    targets = {}
    grids = {}
    for task in tasks:
        if task in ("mu", "alpha"):
            continue  # these come from qm9s.pt
        t, g = load_spectral_tensor(data_dir, task, max_rows=max_molecules, dtype=dtype)
        targets[task] = t
        grids[task] = g
    return targets, grids


def normalize_spectra(spectra, norm="zscore", stats=None):
    """Normalize spectral intensities.

    Args:
        spectra: [N, bins] tensor
        norm: "zscore" or "minmax" or "none"
        stats: optional dict with "mean", "std" for zscore, or "min", "max" for minmax.
               If None, computed from data.

    Returns:
        normalized: [N, bins] tensor
        stats: dict with normalization parameters
    """
    if norm == "none":
        return spectra, {}

    if norm == "zscore":
        if stats is None:
            mean = spectra.mean()
            std = spectra.std()
            if std < 1e-8:
                std = 1.0
        else:
            mean = stats["mean"]
            std = stats["std"]
        stats_out = {"mean": float(mean), "std": float(std), "norm": "zscore"}
        return (spectra - mean) / std, stats_out

    elif norm == "minmax":
        if stats is None:
            vmin = spectra.min()
            vmax = spectra.max()
        else:
            vmin = stats["min"]
            vmax = stats["max"]
        rng = vmax - vmin
        if rng < 1e-8:
            rng = 1.0
        stats_out = {"min": float(vmin), "max": float(vmax), "norm": "minmax"}
        return (spectra - vmin) / rng, stats_out

    else:
        raise ValueError(f"Unknown normalization: {norm}")


def get_spectral_grid(task_name, dtype=torch.float32):
    """Get the spectral grid for a task as a tensor.

    Grid values are reconstructed from known QM9S Figshare specs.
    """
    info = SPECTRAL_GRID.get(task_name)
    if info is None:
        raise ValueError(f"Unknown spectral task: {task_name}")
    return torch.linspace(info["start"], info["end"], info["bins"], dtype=dtype)
