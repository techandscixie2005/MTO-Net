#!/usr/bin/env python3
"""Collect and summarize all MTO-Net results into a final report."""

import json
import os
import sys
from glob import glob


def main():
    project_root = os.path.join(os.path.dirname(__file__), "..")
    os.chdir(project_root)

    report = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "metrics": {},
        "figures": [],
        "checkpoints": [],
        "tests": {},
    }

    # Collect metrics
    for path in glob("outputs/metrics/*.json"):
        key = os.path.basename(path)
        try:
            with open(path) as f:
                report["metrics"][key] = json.load(f)
        except Exception:
            report["metrics"][key] = "(error reading)"

    # Collect figures
    for path in glob("outputs/figures/**/*.png", recursive=True):
        report["figures"].append(path)

    # Collect checkpoints
    for path in glob("outputs/checkpoints/**/best.pt", recursive=True):
        report["checkpoints"].append(path)

    # Collect test results
    for path in glob("outputs/metrics/smoke_mto_forward.json"):
        try:
            with open(path) as f:
                report["tests"]["smoke_mto_forward"] = json.load(f)
        except Exception:
            pass

    out_path = "outputs/reports/final_summary.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Summary written to {out_path}")
    print(f"  Metrics files: {len(report[metrics])}")
    print(f"  Figures: {len(report[figures])}")
    print(f"  Checkpoints: {len(report[checkpoints])}")


if __name__ == "__main__":
    main()
