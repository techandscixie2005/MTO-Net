#!/usr/bin/env python3
"""Audit DetaNet tensor shapes and save a report."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mto.compat import *
from src.mto.detanet_adapter import DetaNetBackboneAdapter
import torch


def main():
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    configs = [
        {"num_features": 128, "maxl": 1, "num_block": 1, "rc": 5.0},
        {"num_features": 128, "maxl": 3, "num_block": 3, "rc": 5.0},
    ]

    shapes = {}
    for cfg in configs:
        key = f"maxl{cfg[maxl]}_blocks{cfg[num_block]}"
        print(f"\n=== DetaNet {key} ===")
        adapter = DetaNetBackboneAdapter(**cfg)

        z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=torch.long)
        pos = torch.randn(9, 3)
        batch = torch.zeros(9, dtype=torch.long)

        with torch.no_grad():
            out = adapter(z=z, pos=pos, batch=batch)

        S = out["atom_features"]
        T = out["atom_tensors"]["T"]
        h0 = out["atom_tensors"]["h0"]

        shapes[key] = {
            "num_features": cfg["num_features"],
            "maxl": cfg["maxl"],
            "num_block": cfg["num_block"],
            "rc": cfg["rc"],
            "S_shape": list(S.shape),
            "T_shape": list(T.shape),
            "h0_shape": list(h0.shape),
            "vdim": adapter.vdim,
            "num_params": sum(p.numel() for p in adapter.parameters()),
        }
        print(f"  S: {list(S.shape)}, T: {list(T.shape)}, vdim: {adapter.vdim}")
        print(f"  Params: {shapes[key][num_params]:,}")

    with open("outputs/metrics/detanet_tensor_shapes.json", "w") as f:
        json.dump(shapes, f, indent=2)

    # Markdown report
    lines = ["# DetaNet Tensor Audit\n"]
    lines.append("## Hook Point\n")
    lines.append("MTO hooks into DetaNet after the Interaction_Block loop.\n")
    lines.append("`DetaNet(out_type=latent, scale=None)` returns `(S, T)` where:\n")
    lines.append("- `S` [N, num_features]: per-atom invariant scalar features\n")
    lines.append("- `T` [N, vdim]: per-atom equivariant irrep tensor\n")
    lines.append("\n## Shape Table\n")
    lines.append("| Config | S shape | T shape | vdim | Params |")
    lines.append("|--------|---------|---------|------|--------|")
    for key, info in shapes.items():
        lines.append(f"| {key} | {info[S_shape]} | {info[T_shape]} | {info[vdim]} | {info[num_params]:,} |")
    lines.append("\n## MTO Usage\n")
    lines.append("- MTO routing uses **only** S (scalar invariant features)\n")
    lines.append("- T is passed through for compatibility but NOT used for routing\n")
    lines.append("- This ensures c_ki coefficients remain scalar/invariant\n")

    with open("outputs/reports/detanet_tensor_audit.md", "w") as f:
        f.write("\n".join(lines))
    print("\nReport saved: outputs/reports/detanet_tensor_audit.md")


if __name__ == "__main__":
    main()
