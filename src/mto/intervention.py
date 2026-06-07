"""Slot intervention analysis for MTO-Net.

Mask one MTO slot at a time and measure prediction deltas.
"""

import torch
import numpy as np


def slot_intervention(
    model, z, pos, batch, slot_idx: int,
) -> dict:
    """Zero out MTO slot k and measure delta in all property predictions.

    Returns:
        dict with keys: baseline, intervention, delta, slot_idx
    """
    model.eval()
    with torch.no_grad():
        # Baseline
        out_base = model(z=z, pos=pos, batch=batch, return_mto=True)

        # Intervention
        out_int = model(z=z, pos=pos, batch=batch, return_mto=True, mask_slot=slot_idx)

    deltas = {}
    for key in out_base:
        if key in ("O", "coeff", "mask", "atom_mask", "K_per_mol", "atom_tensors"):
            continue
        if isinstance(out_base[key], torch.Tensor):
            delta = (out_int[key] - out_base[key]).abs()
            deltas[key] = float(delta.mean().item())

    return {
        "slot_idx": slot_idx,
        "deltas": deltas,
    }


def slot_intervention_sweep(
    model, loader, max_slots: int = None, device: torch.device = None,
) -> list[dict]:
    """Run slot intervention sweep across multiple molecules.

    Returns list of dicts with per-slot deltas averaged over batch.
    """
    model.eval()
    all_results = []

    if device is None:
        device = next(model.parameters()).device

    for batch in loader:
        z = batch["z"].to(device)
        pos = batch["pos"].to(device)
        batch_idx = batch.get("batch", torch.zeros(len(z), dtype=torch.long)).to(device)

        with torch.no_grad():
            out = model(z=z, pos=pos, batch=batch_idx, return_mto=True)
            K_max = out.get("K_per_mol")
            if K_max is not None:
                k_limit = int(K_max.max().item()) if max_slots is None else min(max_slots, int(K_max.max().item()))
            else:
                k_limit = max_slots or 5

        for k in range(k_limit):
            result = slot_intervention(model, z, pos, batch_idx, k)
            result["mol_info"] = {k: v.tolist() if isinstance(v, torch.Tensor) else v
                                  for k, v in batch.items() if k in ("smiles", "mol_id")}
            all_results.append(result)

    return all_results


def intervention_summary(results: list[dict]) -> dict:
    """Aggregate per-slot intervention results."""
    by_slot = {}
    for r in results:
        k = r["slot_idx"]
        if k not in by_slot:
            by_slot[k] = {"deltas": {tk: [] for tk in r["deltas"]}, "count": 0}
        by_slot[k]["count"] += 1
        for tk, dv in r["deltas"].items():
            by_slot[k]["deltas"][tk].append(dv)

    summary = {}
    for k, v in by_slot.items():
        summary[str(k)] = {
            "count": v["count"],
            "deltas": {tk: float(np.mean(dvs)) for tk, dvs in v["deltas"].items()},
        }
    return summary
