"""Plotting utilities for spectral predictions: IR, Raman, UV spectra.

Generates debug figures only. Not for Nature-style final figures.
"""
import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_predicted_vs_target(pred, target, grid, task_name, save_path,
                             mol_idx=0, title=None):
    """Plot predicted vs target spectrum for a single molecule.

    Args:
        pred: [B, bins] or [bins] tensor
        target: [B, bins] or [bins] tensor
        grid: [bins] tensor of x-axis values
        task_name: "ir", "raman", or "uv"
        save_path: output file path
        mol_idx: which molecule to plot
        title: optional title
    """
    if pred.dim() > 1:
        pred = pred[mol_idx]
    if target.dim() > 1:
        target = target[mol_idx]

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    grid_np = grid.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grid_np, target_np, label="Target", alpha=0.7, linewidth=1.5)
    ax.plot(grid_np, pred_np, label="Predicted", alpha=0.7, linewidth=1.5,
            linestyle="--")
    ax.set_xlabel(_xlabel(task_name))
    ax.set_ylabel("Intensity (normalized)")
    ax.set_title(title or f"{task_name.upper()} Spectrum - Molecule {mol_idx}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_spectral_grid(preds_dict, targets_dict, grids, save_dir,
                       mol_indices=None):
    """Plot spectra for multiple tasks in a grid layout.

    Args:
        preds_dict: {task_name: [B, bins] tensor}
        targets_dict: {task_name: [B, bins] tensor}
        grids: {task_name: [bins] tensor}
        save_dir: directory for output figures
        mol_indices: list of molecule indices to plot
    """
    tasks = [t for t in preds_dict if t in targets_dict and t in grids]
    if not mol_indices:
        mol_indices = [0]
    if len(mol_indices) > 8:
        mol_indices = mol_indices[:8]

    n_cols = len(tasks)
    n_rows = len(mol_indices)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows),
                             squeeze=False)

    for r, mol_idx in enumerate(mol_indices):
        for c, task in enumerate(tasks):
            ax = axes[r, c]
            pred = preds_dict[task][mol_idx].detach().cpu().numpy()
            target = targets_dict[task][mol_idx].detach().cpu().numpy()
            grid = grids[task].detach().cpu().numpy()
            ax.plot(grid, target, label="Target", alpha=0.7, linewidth=1.0)
            ax.plot(grid, pred, label="Pred", alpha=0.7, linewidth=1.0,
                    linestyle="--")
            ax.set_title(f"{task.upper()} - Mol {mol_idx}")
            ax.set_xlabel(_xlabel(task))
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "spectral_predictions_grid.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_slot_intervention_spectra(original_pred, intervened_preds, grid,
                                   task_name, save_path, mol_idx=0):
    """Plot original spectrum vs slot-intervened spectra.

    Args:
        original_pred: [bins] tensor, original prediction
        intervened_preds: dict[int -> [bins] tensor], slot_idx -> intervened pred
        grid: [bins] tensor
        task_name: "ir", "raman", or "uv"
        save_path: output file path
        mol_idx: molecule index for title
    """
    if original_pred.dim() > 1:
        original_pred = original_pred[mol_idx]

    original_np = original_pred.detach().cpu().numpy()
    grid_np = grid.detach().cpu().numpy()

    n_slots = len(intervened_preds)
    n_cols = min(4, n_slots)
    n_rows = (n_slots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows),
                             squeeze=False)

    for idx, (slot_k, slot_pred) in enumerate(sorted(intervened_preds.items())):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r, c]
        slot_np = slot_pred[mol_idx].detach().cpu().numpy() if slot_pred.dim() > 1 else slot_pred.detach().cpu().numpy()
        ax.plot(grid_np, original_np, label="Original", alpha=0.6, linewidth=1.0)
        ax.plot(grid_np, slot_np, label=f"Slot {slot_k} zeroed", alpha=0.8,
                linewidth=1.0, linestyle="--")
        ax.set_title(f"Slot {slot_k} Intervention")
        ax.set_xlabel(_xlabel(task_name))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_slots, n_rows * n_cols):
        r, c = idx // n_cols, idx % n_cols
        axes[r, c].set_visible(False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _xlabel(task_name):
    labels = {
        "ir": "Wavenumber (cm$^{-1}$)",
        "raman": "Wavenumber (cm$^{-1}$)",
        "uv": "Energy (eV)",
    }
    return labels.get(task_name, "Grid index")


def setup_mpl_style():
    plt.style.use("default")
    matplotlib.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.dpi": 150,
    })
