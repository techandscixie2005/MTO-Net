"""Frozen probe analysis: test MTO reusability across tasks."""
import torch
import torch.nn as nn
from copy import deepcopy


def freeze_backbone_and_mto(model):
    """Freeze DetaNet backbone and MTO module, keep readout heads trainable."""
    for name, param in model.named_parameters():
        if "readout" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def frozen_probe_train(model, train_loader, val_loader, new_tasks,
                       epochs=20, lr=1e-3, device=None):
    """Train only readout heads on new tasks with frozen MTO."""
    if device is None:
        device = next(model.parameters()).device
    model = deepcopy(model)
    freeze_backbone_and_mto(model)

    # Add new readout heads if needed
    for task_name, out_dim in new_tasks.items():
        if task_name not in model.readout.heads:
            model.readout.heads[task_name] = nn.Sequential(
                nn.Linear(model.feature_dim, 128),
                nn.SiLU(),
                nn.Linear(128, out_dim),
            ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    history = {"train_loss": [], "val_loss": []}
    mse = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            optimizer.zero_grad()
            z = batch["z"].to(device)
            pos = batch["pos"].to(device)
            batch_idx = batch.get("batch", torch.zeros(len(z), dtype=torch.long, device=device)).to(device)
            out = model(z=z, pos=pos, batch=batch_idx)
            loss = 0.0
            for task in new_tasks:
                if task in batch:
                    t = batch[task].to(device).float()
                    if task in out:
                        loss = loss + mse(out[task], t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        train_loss = total_loss / max(n, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                z = batch["z"].to(device)
                pos = batch["pos"].to(device)
                batch_idx = batch.get("batch", torch.zeros(len(z), dtype=torch.long, device=device)).to(device)
                out = model(z=z, pos=pos, batch=batch_idx)
                for task in new_tasks:
                    if task in batch:
                        t = batch[task].to(device).float()
                        if task in out:
                            val_loss = val_loss + mse(out[task], t).item()
                            n_val += 1
        val_loss = val_loss / max(n_val, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    return {"history": history, "model": model, "frozen_params": sum(
        1 for p in model.parameters() if not p.requires_grad
    )}


def compare_probe_modes(frozen_result, finetune_result, from_scratch_result):
    """Compare frozen probe vs fine-tuning vs from-scratch.

    Returns dict comparing validation losses.
    """
    return {
        "frozen_val_loss": frozen_result["history"]["val_loss"][-1],
        "finetune_val_loss": finetune_result["history"]["val_loss"][-1] if finetune_result else None,
        "from_scratch_val_loss": from_scratch_result["history"]["val_loss"][-1] if from_scratch_result else None,
        "frozen_params_count": frozen_result["frozen_params"],
    }
