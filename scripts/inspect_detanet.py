#!/usr/bin/env python3
"""Inspect DetaNet backbone and identify atom-level hidden representations."""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mto.compat import *  # noqa

DETANET_DIR = os.path.join("third_party", "DetaNet")
sys.path.insert(0, DETANET_DIR)

print("=== DetaNet Backbone Inspection ===\n")

# File listing
print("--- Files under detanet_model/ ---")
for root, dirs, files in os.walk(os.path.join(DETANET_DIR, "detanet_model")):
    for f in files:
        if f.endswith(".py"):
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            print(f"  {fpath} ({size} bytes)")
print()

# Import and test
try:
    import torch
    from detanet_model.detanet import DetaNet
    
    print("[OK] DetaNet class imported")
    
    model = DetaNet(
        num_features=128, maxl=3, num_block=3, rc=5.0,
        out_type="latent", summation=False, scalar_outsize=0,
        irreps_out=None, scale=None,
        device=torch.device("cpu"),
    )
    print(f"[OK] DetaNet() instantiated (out_type=latent)")
    
    for name, module in model.named_children():
        print(f"     child: {name} -> {type(module).__name__}")
    
    z = torch.tensor([6, 6, 8, 1, 1, 1], dtype=torch.long)
    pos = torch.randn(6, 3)
    batch = torch.zeros(6, dtype=torch.long)
    
    with torch.no_grad():
        S, T = model(z=z, pos=pos, batch=batch)
    
    print(f"\n[OK] Forward pass success")
    print(f"  S (scalar features): shape={list(S.shape)}, dtype={S.dtype}")
    print(f"  T (irrep tensor):    shape={list(T.shape)}, dtype={T.dtype}")
    print(f"  num_features: {model.features}")
    print(f"  vdim (irrep dim): {model.vdim}")
    
    inspection = {
        "model_type": type(model).__name__,
        "num_features": model.features,
        "vdim": model.vdim,
        "S_shape": list(S.shape),
        "T_shape": list(T.shape),
        "hook_location": "after Interaction_Block loop (out_type=latent returns S, T)",
        "hook_description": (
            "DetaNet.forward with out_type=latent returns (S, T) where "
            "S is per-atom scalar features [N, num_features] and "
            "T is per-atom irrep tensor [N, vdim]. "
            "S is used as atom_features for the MTO module."
        ),
    }
    
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/logs/detanet_inspection.json", "w") as f:
        json.dump(inspection, f, indent=2, default=str)
    with open("outputs/metrics/detanet_tensor_shapes.json", "w") as f:
        json.dump(inspection, f, indent=2)
    print(f"\nInspection saved")
    
except Exception as e:
    import traceback
    print(f"[FAIL] DetaNet inspection failed:")
    traceback.print_exc()
