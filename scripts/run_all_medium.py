#!/usr/bin/env python3
"""Run medium subset training pipeline."""

import json
import os
import sys
import subprocess
from datetime import datetime


def run_step(name, cmd):
    print(f"\n{=*60}")
    print(f"STEP: {name}")
    print(f"Command: { .join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    project_root = os.path.join(os.path.dirname(__file__), "..")
    os.chdir(project_root)

    os.makedirs("outputs/reports", exist_ok=True)
    config = {"num_mols": 512}

    # Step 1: Create medium subset
    ok = run_step("Create medium subset",
                  [sys.executable, "scripts/make_qm9s_subset.py",
                   "--name", "medium",
                   "--num-mols", str(config["num_mols"]),
                   "--seed", "0"])
    if not ok:
        print("FAIL: Cannot create medium subset")
        return

    # Step 2: Prepare processed data
    ok = run_step("Prepare medium data",
                  [sys.executable, "scripts/prepare_qm9s.py",
                   "--data-dir", "data/qm9s/subset_medium",
                   "--out", "data/qm9s/subset_medium/processed"])
    if not ok:
        print("FAIL: Cannot prepare medium data")
        return

    # Step 3: Train Stage A (3 seeds)
    for seed in [0, 1, 2]:
        ok = run_step(f"Stage A seed {seed}",
                      [sys.executable, "scripts/train_stage.py",
                       "--stage", "stage_a",
                       "--data-dir", "data/qm9s/subset_medium/processed",
                       "--epochs", "50",
                       "--seed", str(seed),
                       "--batch-size", "8",
                       "--checkpoint-dir", "outputs/checkpoints"])
        if not ok:
            print(f"FAIL: Stage A seed {seed}")
            return

    # Step 4: Train Stage B (transfer from seed 0)
    ok = run_step("Stage B transfer",
                  [sys.executable, "scripts/train_stage.py",
                   "--stage", "stage_b",
                   "--data-dir", "data/qm9s/subset_medium/processed",
                   "--epochs", "100",
                   "--seed", "0",
                   "--batch-size", "8",
                   "--checkpoint-dir", "outputs/checkpoints"])
    if not ok:
        print("FAIL: Stage B")

    # Step 5: Train Stage C (transfer from seed 0)
    ok = run_step("Stage C transfer",
                  [sys.executable, "scripts/train_stage.py",
                   "--stage", "stage_c",
                   "--data-dir", "data/qm9s/subset_medium/processed",
                   "--epochs", "100",
                   "--seed", "0",
                   "--batch-size", "8",
                   "--checkpoint-dir", "outputs/checkpoints"])
    if not ok:
        print("FAIL: Stage C")

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "completed": True,
    }
    with open("outputs/reports/medium_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nMedium pipeline complete")


if __name__ == "__main__":
    main()
