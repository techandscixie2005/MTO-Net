"""Test spectrum plotting: predicted vs target figure generation."""
import os, sys, pytest, tempfile
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_plot_predicted_vs_target():
    """Should generate a spectrum comparison figure."""
    from src.mto.plotting.plot_spectra import plot_predicted_vs_target
    pred = torch.randn(100) * 0.5 + 1.0
    target = torch.randn(100) * 0.5 + 1.0
    grid = torch.linspace(500, 4000, 100)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_ir_pred.png")
        plot_predicted_vs_target(pred, target, grid, "ir", path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000  # non-empty PNG


def test_plot_slot_intervention_spectra():
    """Should generate slot intervention figure."""
    from src.mto.plotting.plot_spectra import plot_slot_intervention_spectra
    original = torch.randn(100) * 0.5 + 1.0
    grid = torch.linspace(500, 4000, 100)
    intervened = {
        0: original * 0.8 + torch.randn(100) * 0.1,
        1: original * 0.9 + torch.randn(100) * 0.1,
        3: original * 0.7 + torch.randn(100) * 0.1,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_intervention.png")
        plot_slot_intervention_spectra(original, intervened, grid, "ir", path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000


def test_plot_spectral_grid():
    """Should generate multi-molecule, multi-task grid figure."""
    from src.mto.plotting.plot_spectra import plot_spectral_grid
    preds = {
        "ir": torch.randn(4, 100),
        "raman": torch.randn(4, 100),
    }
    targets = {
        "ir": torch.randn(4, 100),
        "raman": torch.randn(4, 100),
    }
    grids = {
        "ir": torch.linspace(500, 4000, 100),
        "raman": torch.linspace(500, 4000, 100),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_spectral_grid(preds, targets, grids, tmpdir, mol_indices=[0, 1])
        path = os.path.join(tmpdir, "spectral_predictions_grid.png")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000


def test_uv_plotting():
    """UV spectra should use eV on x-axis."""
    from src.mto.plotting.plot_spectra import plot_predicted_vs_target
    pred = torch.randn(50) * 0.5 + 0.5
    target = torch.randn(50) * 0.5 + 0.5
    grid = torch.linspace(1.0, 15.0, 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_uv_pred.png")
        plot_predicted_vs_target(pred, target, grid, "uv", path)
        assert os.path.exists(path)
