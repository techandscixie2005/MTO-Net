#!/usr/bin/env python3
"""Slot intervention analysis: mask one MTO slot, measure prediction delta."""

import argparse, json, os, sys
import torch, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.mto.compat import *
from src.mto.mto_model import MTONet
from src.mto.intervention import slot_intervention_sweep, intervention_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="stage_c")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-mols", type=int, default=256)
    parser.add_argument("--data-dir", default="data/qm9s/processed")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(args.checkpoint_dir, f"{args.stage}_seed0", "best.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: {ckpt_path} not found")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tasks = ckpt.get("tasks", {k: 3 for k in ["mu", "alpha"]})
    model = MTONet(tasks=tasks).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_path = os.path.join(args.data_dir, f"{args.split}.pt")
    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found")
        return

    test_data = torch.load(test_path, map_location="cpu", weights_only=False)
    limit = min(args.num_mols, len(test_data))
    test_data = test_data[:limit]
    print(f"Running slot intervention on {limit} molecules")

    from torch.utils.data import DataLoader
    from src.mto.dataset_qm9s import collate_fn

    class ListDS(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, i):
            return self.data[i]

    loader = DataLoader(ListDS(test_data), batch_size=8, collate_fn=collate_fn)
    results = slot_intervention_sweep(model, loader, max_slots=10, device=device)
    summary = intervention_summary(results)

    os.makedirs(f"{args.out_dir}/metrics", exist_ok=True)
    os.makedirs(f"{args.out_dir}/figures/stability", exist_ok=True)

    with open(f"{args.out_dir}/metrics/slot_intervention.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(f"{args.out_dir}/metrics/slot_intervention.csv", "w") as f:
        slots = sorted(summary.keys(), key=lambda x: int(x))
        tasks_list = list(summary[slots[0]]["deltas"].keys())
        f.write("slot," + ",".join(tasks_list) + "\n")
        for k in slots:
            deltas = summary[k]["deltas"]
            f.write(k + "," + ",".join(f"{deltas.get(t, 0):.6f}" for t in tasks_list) + "\n")

    print("Slot intervention summary (top slots):")
    for k in sorted(summary.keys(), key=lambda x: int(x))[:5]:
        deltas = summary[k]["deltas"]
        parts = [f"{t}={deltas.get(t, 0):.4f}" for t in deltas]
        print(f"  Slot {k}: " + ", ".join(parts))
    print(f"Saved: {args.out_dir}/metrics/slot_intervention.csv")

if __name__ == "__main__":
    main()
