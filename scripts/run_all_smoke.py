#!/usr/bin/env python3
"""Run the complete smoke test pipeline.

Steps:
  1. Verify MTO forward pass (synthetic molecules)
  2. Inspect DetaNet backbone
  3. Run MTO forward with DetaNet backbone
  4. Save all outputs
"""

import json
import os
import sys
import subprocess


def run_step(name, cmd):
    print(f"\n{=*60}")
    print(f"STEP: {name}")
    print(f"{=*60}")
    print(f"Command: { .join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-500:] if result.stdout else "(no stdout)")
    if result.stderr:
        print("STDERR:", result.stderr[-200:])
    return result.returncode == 0


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/figures/mto_maps", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    results = {}
    all_pass = True

    # Step 1: Synthetic smoke test
    ok = run_step("Smoke MTO forward (synthetic)",
                  [sys.executable, "scripts/smoke_mto_forward.py"])
    results["smoke_synthetic"] = {"pass": ok}
    if not ok:
        all_pass = False
        print("FAIL: Synthetic smoke test")
        return

    # Step 2: DetaNet inspection
    ok = run_step("Inspect DetaNet",
                  [sys.executable, "scripts/inspect_detanet.py"])
    results["inspect_detanet"] = {"pass": ok}
    if not ok:
        all_pass = False

    # Step 3: Smoke with DetaNet backbone
    ok = run_step("Smoke MTO forward (DetaNet)",
                  [sys.executable, "scripts/smoke_mto_forward.py", "--use-detanet"])
    results["smoke_detanet"] = {"pass": ok}
    if not ok:
        all_pass = False

    # Step 4: MTO maps (from synthetic model)
    ok = run_step("MTO maps (synthetic model)",
                  [sys.executable, "scripts/plot_one_mto_map.py"])
    results["mto_maps"] = {"pass": ok}
    if not ok:
        print("WARN: MTO maps failed (non-critical)")

    # Save report
    report = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "results": results,
        "all_pass": all_pass,
    }
    with open("outputs/reports/smoke_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{=*60}")
    print(f"SMOKE TEST COMPLETE")
    print(f"{=*60}")
    print(f"Results:")
    for step, r in results.items():
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {step}: {status}")
    print(f"\nOverall: {PASS if all_pass else FAIL}")
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
