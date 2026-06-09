"""Test spectral CSV alignment to qm9s.pt molecules."""
import os, sys, pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "qm9s")


@pytest.fixture(scope="module")
def spectral_index():
    from src.mto.dataset_qm9s import load_spectral_index
    return load_spectral_index(DATA_DIR, ["ir"])


def test_row_count_matches_pt(spectral_index):
    """CSV row count should be within 1 of qm9s.pt count (known 1-molecule discrepancy)."""
    n_csv = len(spectral_index["ir"])
    n_pt = 129818  # known from dataset audit
    diff = abs(n_csv - n_pt)
    assert diff <= 1, (
        f"CSV has {n_csv} molecules, PT has {n_pt} (diff={diff}). "
        f"If diff > 1, alignment strategy needs review."
    )


def test_index_0_exists(spectral_index):
    """Molecule index 0 should exist in spectral data."""
    assert 0 in spectral_index["ir"]


def test_index_max(spectral_index):
    """Max index should be near total count."""
    max_idx = max(spectral_index["ir"].keys())
    n_csv = len(spectral_index["ir"])
    assert max_idx < n_csv + 2  # allow for the known 1-molecule gap


def test_first_spectrum_nonzero(spectral_index):
    """First molecule spectrum should have non-zero values."""
    spec = spectral_index["ir"][0]
    non_zero = sum(1 for v in spec if v != 0.0)
    assert non_zero > 0, "First IR spectrum is all zeros"


def test_all_spectra_3501_bins(spectral_index):
    """All IR spectra should have 3501 bins."""
    for mol_id, spec in list(spectral_index["ir"].items())[:100]:
        assert len(spec) == 3501, f"Mol {mol_id}: expected 3501 bins, got {len(spec)}"


def test_positive_intensities(spectral_index):
    """IR intensities should be non-negative (IR/Raman are intensity spectra)."""
    for mol_id, spec in list(spectral_index["ir"].items())[:50]:
        for v in spec[:10]:  # check first 10 bins
            assert v >= 0, f"Mol {mol_id}: negative IR intensity {v}"


def test_spectral_collator(spectral_index):
    """SpectralCollator should attach spectral targets by mol_id."""
    from src.mto.dataset_qm9s import SpectralCollator, load_qm9s_raw, QM9SDataset

    pt_path = os.path.join(DATA_DIR, "qm9s.pt")
    if not os.path.exists(pt_path):
        pytest.skip("qm9s.pt not found")
    data = load_qm9s_raw(pt_path)
    dataset = QM9SDataset(data[:16])
    collator = SpectralCollator(spectral_index, ["ir"], downsample=50)
    batch = collator([dataset[i] for i in range(8)])
    assert "ir" in batch
    assert batch["ir"].shape == (8, 50)  # downsampled
    assert batch["z"].ndim == 1
    assert batch["pos"].ndim == 2


def test_downsample_collator(spectral_index):
    """SpectralCollator with downsample should produce fewer bins."""
    from src.mto.dataset_qm9s import SpectralCollator, load_qm9s_raw, QM9SDataset

    pt_path = os.path.join(DATA_DIR, "qm9s.pt")
    if not os.path.exists(pt_path):
        pytest.skip("qm9s.pt not found")
    data = load_qm9s_raw(pt_path)
    dataset = QM9SDataset(data[:16])
    collator = SpectralCollator(spectral_index, ["ir"], downsample=100)
    batch = collator([dataset[i] for i in range(8)])
    assert batch["ir"].shape == (8, 100)
