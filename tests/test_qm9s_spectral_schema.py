"""Test QM9S spectral CSV schema: verify file existence, format, alignment.

Tests run without GPU and without loading full CSVs into memory.
"""
import csv, os, sys, pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "qm9s")


def _boraden_path(task):
    """Get path with real boraden filename."""
    names = {"ir": "ir_boraden.csv", "raman": "raman_boraden.csv", "uv": "uv_boraden.csv"}
    return os.path.join(DATA_DIR, names[task])


@pytest.mark.parametrize("task", ["ir", "raman", "uv"])
def test_spectral_file_exists(task):
    """Each spectral boraden CSV must exist. Skip locally, require on HPC."""
    path = _boraden_path(task)
    if not os.path.exists(path):
        pytest.skip(f"Spectral file not found: {path} — expected on HPC server only")


@pytest.mark.parametrize("task,expected_bins", [
    ("ir", 3501), ("raman", 3501), ("uv", 701),
])
def test_spectral_header_bins(task, expected_bins):
    """Verify expected number of spectral bins from header."""
    path = _boraden_path(task)
    if not os.path.exists(path):
        pytest.skip(f"Spectral file not found: {path}")
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
    n_bins = len(header) - 1  # skip mol_id column
    assert n_bins == expected_bins, f"{task}: expected {expected_bins} bins, got {n_bins}"


def test_spectral_grid_values_ir():
    """IR spectral grid should be 500-4000 cm-1."""
    path = _boraden_path("ir")
    if not os.path.exists(path):
        pytest.skip("IR file missing")
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
    grid_start = float(header[1])
    grid_end = float(header[-1])
    assert 499 <= grid_start <= 501, f"IR grid start {grid_start} not near 500"
    assert 3999 <= grid_end <= 4001, f"IR grid end {grid_end} not near 4000"


def test_spectral_grid_values_uv():
    """UV spectral grid should be 1-15 eV."""
    path = _boraden_path("uv")
    if not os.path.exists(path):
        pytest.skip("UV file missing")
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
    grid_start = float(header[1])
    grid_end = float(header[-1])
    assert 0.9 <= grid_start <= 1.1, f"UV grid start {grid_start} not near 1.0"
    assert 14.5 <= grid_end <= 15.5, f"UV grid end {grid_end} not near 15.0"


def test_row_count_consistency():
    """All three spectral CSVs should have the same number of data rows."""
    counts = {}
    for task in ["ir", "raman", "uv"]:
        path = _boraden_path(task)
        if not os.path.exists(path):
            count = None
        else:
            with open(path, "rb") as f:
                count = sum(1 for _ in f) - 1  # exclude header
        counts[task] = count

    valid = [c for c in counts.values() if c is not None]
    if len(valid) < 2:
        pytest.skip("Not enough spectral files to compare")
    assert len(set(valid)) == 1, f"Row counts differ: {counts}"


def test_no_empty_spectral_row():
    """First 100 IR rows should have non-zero intensities."""
    path = _boraden_path("ir")
    if not os.path.exists(path):
        pytest.skip("IR file missing")
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if i >= 100:
                break
            vals = [float(x) for x in row[1:]]
            assert any(v != 0.0 for v in vals), f"IR row {i} is all zeros"


def test_spectral_index_builds():
    """load_spectral_index should build valid indices."""
    if not os.path.exists(_boraden_path("ir")):
        pytest.skip("Spectral CSV files not available locally")
    from src.mto.dataset_qm9s import load_spectral_index
    idx = load_spectral_index(DATA_DIR, ["ir"])
    assert "ir" in idx
    assert len(idx["ir"]) > 0
    # First molecule should be index 0
    assert 0 in idx["ir"]
    assert len(idx["ir"][0]) == 3501


def test_real_filenames_used():
    """Code must reference boraden not broaden filenames."""
    import src.mto.data.qm9s_spectral as qs
    for task, fname in qs.SPECTRAL_FILES.items():
        assert "boraden" in fname, f"Spectral file for {task} must use 'boraden': {fname}"
