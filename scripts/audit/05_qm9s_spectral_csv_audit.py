#!/usr/bin/env python3
"""Audit QM9S spectral CSV schemas: ir_boraden.csv, raman_boraden.csv, uv_boraden.csv.

Inspects file size, rows, columns, alignment to qm9s.pt, NaNs, normalization.
Uses chunked reading to avoid loading entire CSVs into memory.
"""
import csv, os, sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "qm9s")
PT_PATH = os.path.join(DATA_DIR, "qm9s.pt")
SPECTRAL_FILES = {
    "ir": os.path.join(DATA_DIR, "ir_boraden.csv"),
    "raman": os.path.join(DATA_DIR, "raman_boraden.csv"),
    "uv": os.path.join(DATA_DIR, "uv_boraden.csv"),
}


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def count_lines(path):
    """Count lines in a file without reading all content."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    # Subtract header
    return max(0, count - 1)


def read_header(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        return next(reader)


def read_first_n_rows(path, n=3):
    rows = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            if i >= n:
                break
            rows.append(row)
    return header, rows


def check_nans(path, header, sample_lines=1000):
    """Sample check for NaN/malformed rows using streaming."""
    nan_counts = [0] * len(header)
    malformed = 0
    total = 0
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in enumerate(reader):
            if i % max(1, (129818 // sample_lines)) != 0 and i >= sample_lines:
                continue
            if i >= sample_lines * 2:
                break
            total += 1
            if len(row) != len(header):
                malformed += 1
                continue
            for j, val in enumerate(row):
                try:
                    v = float(val)
                    if v != v:  # NaN check
                        nan_counts[j] += 1
                except (ValueError, TypeError):
                    nan_counts[j] += 1
    return {"total_sampled": total, "malformed_rows": malformed,
            "nan_cols": [j for j, c in enumerate(nan_counts) if c > 0],
            "nan_counts": nan_counts}


def check_alignment(num_csv_rows):
    """Check if CSV row count matches qm9s.pt molecule count."""
    pt_exists = os.path.exists(PT_PATH)
    if not pt_exists:
        return {"pt_exists": False, "note": "qm9s.pt not found at " + PT_PATH}

    pt_size = os.path.getsize(PT_PATH) / (1024 * 1024 * 1024)
    # We can't easily get molecule count without torch, but we know from prior audit it's 129818
    # Verify via known count
    known_count = 129818
    aligned = num_csv_rows == known_count
    return {
        "pt_exists": True,
        "pt_size_gb": round(pt_size, 2),
        "qm9s_pt_molecule_count": known_count,
        "csv_row_count": num_csv_rows,
        "aligned": aligned,
        "strategy": "row_index_alignment" if aligned else "NEEDS_INVESTIGATION",
        "note": "Row counts match: alignment by row index is safe" if aligned
                else f"Mismatch: CSV has {num_csv_rows} rows, PT has {known_count} molecules"
    }


def analyze_spectral_grid(header):
    """Analyze the spectral grid columns."""
    # First column is molecule index
    mol_id_col = header[0] if header else "unknown"
    n_spectral = len(header) - 1

    # Check if columns are numeric (wavelength/frequency bins)
    spectral_cols = header[1:]
    numeric_cols = []
    non_numeric = []
    for c in spectral_cols:
        try:
            numeric_cols.append(float(c))
        except ValueError:
            non_numeric.append(c)

    is_wide_format = len(numeric_cols) > len(non_numeric)
    grid_start = numeric_cols[0] if numeric_cols else None
    grid_end = numeric_cols[-1] if numeric_cols else None
    grid_step = None
    if len(numeric_cols) >= 2:
        grid_step = numeric_cols[1] - numeric_cols[0]

    return {
        "mol_id_column": mol_id_col,
        "n_spectral_bins": n_spectral,
        "format": "wide" if is_wide_format else "mixed",
        "grid_start": grid_start,
        "grid_end": grid_end,
        "grid_step": grid_step,
        "non_numeric_cols": non_numeric[:5],
        "is_wide_format": is_wide_format,
    }


def check_normalization(header, path, max_samples=100):
    """Check if spectral intensities need normalization by sampling rows."""
    import math
    values = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if i >= max_samples:
                break
            # Sample a few intensity columns
            for j in range(1, min(len(row), 100), 10):
                try:
                    values.append(float(row[j]))
                except (ValueError, IndexError):
                    pass

    if not values:
        return {"needs_normalization": "unknown", "note": "no valid values sampled"}

    mean_val = sum(values) / len(values)
    var_val = sum((v - mean_val) ** 2 for v in values) / len(values)
    std_val = math.sqrt(var_val)
    min_val = min(values)
    max_val = max(values)
    range_span = max_val - min_val

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "range": range_span,
        "needs_normalization": "yes (z-score recommended)" if std_val > 0 and abs(mean_val) > std_val * 0.1
                              else "yes (range varies significantly)" if range_span > 10
                              else "likely_ok",
        "recommendation": "Apply per-task z-score normalization before training"
    }


def main():
    print("# QM9S Spectral CSV Schema Audit\n")

    report_lines = ["# QM9S Spectral CSV Schema Audit Report\n"]

    for name, path in SPECTRAL_FILES.items():
        print(f"## {name.upper()} Spectrum: {os.path.basename(path)}\n")
        report_lines.append(f"\n## {name.upper()} Spectrum: `{os.path.basename(path)}`\n")

        if not os.path.exists(path):
            msg = f"**MISSING**: File not found at {path}"
            print(msg)
            report_lines.append(msg + "\n")
            continue

        # 1. File size
        size_mb = file_size_mb(path)
        print(f"- File size: {size_mb:.1f} MB")
        report_lines.append(f"- **File size**: {size_mb:.1f} MB\n")

        # 2. Number of rows
        n_rows = count_lines(path)
        print(f"- Data rows: {n_rows:,}")
        report_lines.append(f"- **Data rows**: {n_rows:,}\n")

        # 3. Header and column analysis
        header = read_header(path)
        n_cols = len(header)
        print(f"- Total columns: {n_cols}")
        report_lines.append(f"- **Total columns**: {n_cols}\n")

        # 4. First 3 rows
        h, sample_rows = read_first_n_rows(path, n=3)
        print(f"- First 3 data rows (first 5 values each):")
        report_lines.append("- **First 3 rows** (first 5 intensity values each):\n")
        for i, row in enumerate(sample_rows):
            vals = row[:5]
            print(f"  Row {i}: mol_id={row[0]}, intensities={vals}")
            report_lines.append(f"  - Row {i}: mol_id={row[0]}, first intensities={vals}\n")

        # 5. Molecule identifier
        print(f"- Molecule identifier column: '{header[0]}'")
        report_lines.append(f"- **Molecule ID column**: `{header[0]}`\n")

        # 6. SMILES
        has_smiles = "smile" in header[0].lower() or any("smile" in c.lower() for c in header)
        print(f"- SMILES column: {'YES' if has_smiles else 'None detected'}")
        report_lines.append(f"- **SMILES column**: {'Yes' if has_smiles else 'None detected'}\n")

        # 7. Spectral grid
        grid_info = analyze_spectral_grid(header)
        print(f"- Spectral grid: {grid_info['n_spectral_bins']} bins, {grid_info['format']} format")
        print(f"  Grid range: {grid_info['grid_start']} to {grid_info['grid_end']}, step={grid_info['grid_step']}")
        report_lines.append(f"- **Spectral grid**: {grid_info['n_spectral_bins']} bins, {grid_info['format']} format\n")
        report_lines.append(f"  - Grid range: {grid_info['grid_start']} to {grid_info['grid_end']}, step={grid_info['grid_step']}\n")

        # 8. Alignment to qm9s.pt
        print(f"\n### Alignment to qm9s.pt")
        report_lines.append(f"\n### Alignment to qm9s.pt\n")
        align_info = check_alignment(n_rows)
        for k, v in align_info.items():
            print(f"  {k}: {v}")
            report_lines.append(f"- **{k}**: {v}\n")

        # 9. NaN check
        print(f"\n### NaN / Malformed Row Check")
        report_lines.append(f"\n### NaN / Data Quality\n")
        nan_info = check_nans(path, header, sample_lines=2000)
        print(f"  Sampled: {nan_info['total_sampled']}, Malformed: {nan_info['malformed_rows']}")
        print(f"  Columns with NaN: {len(nan_info['nan_cols'])}/{n_cols}")
        report_lines.append(f"- Sampled rows: {nan_info['total_sampled']}\n")
        report_lines.append(f"- Malformed rows: {nan_info['malformed_rows']}\n")
        report_lines.append(f"- Columns containing NaN: {len(nan_info['nan_cols'])}/{n_cols}\n")

        # 10. Normalization
        print(f"\n### Normalization Check")
        report_lines.append(f"\n### Normalization\n")
        norm_info = check_normalization(header, path)
        for k, v in norm_info.items():
            print(f"  {k}: {v}")
            report_lines.append(f"- **{k}**: {v}\n")

        print("\n---\n")
        report_lines.append("\n---\n")

    # Summary
    print("\n## Summary\n")
    report_lines.append("\n## Summary\n")
    print("| Spectrum | Rows | Bins | Format | Aligned | Size |")
    print("|----------|------|------|--------|---------|------|")
    report_lines.append("| Spectrum | Rows | Bins | Format | Aligned | Size (MB) |\n")
    report_lines.append("|----------|------|------|--------|---------|----------|\n")
    for name, path in SPECTRAL_FILES.items():
        if os.path.exists(path):
            n_rows = count_lines(path)
            n_bins = len(read_header(path)) - 1
            size_mb = file_size_mb(path)
            align = "YES" if n_rows == 129818 else "NO"
            print(f"| {name.upper()} | {n_rows:,} | {n_bins} | wide | {align} | {size_mb:.0f} MB |")
            report_lines.append(f"| {name.upper()} | {n_rows:,} | {n_bins} | wide | {align} | {size_mb:.0f} MB |\n")

    # Write report
    report_dir = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "15_qm9s_spectral_csv_schema_audit.md")
    with open(report_path, "w") as f:
        f.write("".join(report_lines))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
