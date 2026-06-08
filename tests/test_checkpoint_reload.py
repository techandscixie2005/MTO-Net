"""Test checkpoint save and reload for MTO-Net."""
import torch
import sys, os, tempfile
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "third_party", "DetaNet"))
from src.mto.compat import *  # noqa


def test_checkpoint_reload_identity():
    """Model output should be identical after save+load."""
    from src.mto.mto_model import MTONet

    model = MTONet(
        feature_dim=32,
        mto_hidden_dim=32,
        readout_hidden_dim=32,
        tasks={"mu": 3, "alpha": 9},
        detanet_kwargs={"maxl": 1, "num_block": 1, "rc": 5.0},
        use_activity_gate=True,
    )
    model.eval()

    z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    pos = torch.randn(9, 3)
    batch = torch.zeros(9, dtype=torch.long)

    with torch.no_grad():
        out1 = model(z=z, pos=pos, batch=batch)

    tmp = os.path.join(tempfile.gettempdir(), "mto_test_ckpt.pt")
    try:
        torch.save({"model_state_dict": model.state_dict()}, tmp)
        loaded = torch.load(tmp, map_location="cpu", weights_only=False)

        model2 = MTONet(
            feature_dim=32, mto_hidden_dim=32, readout_hidden_dim=32,
            tasks={"mu": 3, "alpha": 9},
            detanet_kwargs={"maxl": 1, "num_block": 1, "rc": 5.0},
            use_activity_gate=True,
        )
        model2.load_state_dict(loaded["model_state_dict"])
        model2.eval()

        with torch.no_grad():
            out2 = model2(z=z, pos=pos, batch=batch)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    assert torch.allclose(out1["mu"], out2["mu"], atol=1e-5), \
        f"mu mismatch: {out1['mu']} vs {out2['mu']}"
    assert torch.allclose(out1["alpha"], out2["alpha"], atol=1e-5)


def test_training_one_step():
    """Model should complete one optimizer step without NaN."""
    from src.mto.mto_model import MTONet
    from src.mto.losses import CompositeLoss

    model = MTONet(
        feature_dim=32, mto_hidden_dim=32, readout_hidden_dim=32,
        tasks={"mu": 3, "alpha": 9},
        detanet_kwargs={"maxl": 1, "num_block": 1, "rc": 5.0},
        use_activity_gate=True,
    )
    model.train()

    z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    pos = torch.randn(9, 3)
    batch = torch.zeros(9, dtype=torch.long)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = CompositeLoss({"mu": 1.0, "alpha": 1.0},
                              diversity_weight=1e-3, entropy_weight=1e-3)

    out = model(z=z, pos=pos, batch=batch, return_mto=True)
    target = {"mu": torch.randn(1, 3), "alpha": torch.randn(1, 9)}
    loss, per_task = criterion(out, target,
                               O=out.get("O"), mask=out.get("mask"),
                               routing_logits=out.get("routing_logits", []))

    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    loss.backward()
    optimizer.step()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert param.grad is not None
            break
    else:
        raise AssertionError("No parameter had gradients")
