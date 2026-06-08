"""Test checkpoint resume and stage initialization."""
import os
import sys
import tempfile
import torch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "third_party", "DetaNet"))
from src.mto.compat import *  # noqa


def _make_model(tasks=None, use_activity_gate=False):
    from src.mto.mto_model import MTONet
    return MTONet(
        feature_dim=32, mto_hidden_dim=32, readout_hidden_dim=32,
        tasks=tasks or {"mu": 3, "alpha": 9},
        detanet_kwargs={"maxl": 1, "num_block": 1, "rc": 5.0},
        use_activity_gate=use_activity_gate,
    )


def test_resume_checkpoint_identity():
    """Model output unchanged after save+reload cycle."""
    model = _make_model()
    model.eval()
    z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    pos = torch.randn(9, 3)
    batch = torch.zeros(9, dtype=torch.long)

    with torch.no_grad():
        out1 = model(z=z, pos=pos, batch=batch)

    tmpdir = tempfile.mkdtemp()
    ckpt_path = os.path.join(tmpdir, "test.pt")
    try:
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model2 = _make_model()
        model2.load_state_dict(ckpt["model_state_dict"])
        model2.eval()
        with torch.no_grad():
            out2 = model2(z=z, pos=pos, batch=batch)
        assert torch.allclose(out1["mu"], out2["mu"], atol=1e-5)
        assert torch.allclose(out1["alpha"], out2["alpha"], atol=1e-5)
    finally:
        os.remove(ckpt_path)
        os.rmdir(tmpdir)


def test_init_from_missing_head_keys():
    """init-from should load backbone+MTO weights, skip new heads cleanly."""
    model_a = _make_model(tasks={"mu": 3, "alpha": 9})
    model_a.eval()

    tmpdir = tempfile.mkdtemp()
    ckpt_path = os.path.join(tmpdir, "stage_a.pt")
    try:
        torch.save({"model_state_dict": model_a.state_dict()}, ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Stage B model has extra heads (ir, raman) not in Stage A
        model_b = _make_model(tasks={"mu": 3, "alpha": 9, "ir": 3501})
        model_b.eval()

        # Try loading with strict=False to handle missing keys
        missing, unexpected = model_b.load_state_dict(
            ckpt["model_state_dict"], strict=False)

        # ir head should be missing (allowed)
        ir_missing = [k for k in missing if "ir" in k or "readout.heads.ir" in k]
        assert len(ir_missing) > 0, f"Expected missing ir keys, got: {missing[:5]}..."
        # backbone keys should load fine
        backbone_missing = [k for k in missing
                           if "backbone" in k or "mto" in k or "route_mlp" in k]
        assert len(backbone_missing) == 0, f"Found unexpected missing backbone keys: {backbone_missing}"
    finally:
        os.remove(ckpt_path)
        os.rmdir(tmpdir)


def test_load_optimizer_state():
    """Checkpoint saves optimizer state, and loading it doesn't crash."""
    model = _make_model(use_activity_gate=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    tmpdir = tempfile.mkdtemp()
    ckpt_path = os.path.join(tmpdir, "opt.pt")
    try:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 5,
            "val_loss": 0.5,
        }, ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 5
        model2 = _make_model(use_activity_gate=False)
        model2.load_state_dict(ckpt["model_state_dict"])
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        opt2.load_state_dict(ckpt["optimizer_state_dict"])
        # Verify optimizer can step
        z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=torch.long)
        pos = torch.randn(9, 3)
        batch = torch.zeros(9, dtype=torch.long)
        out = model2(z=z, pos=pos, batch=batch)
        loss = (out["mu"] ** 2).sum() + (out["alpha"] ** 2).sum()
        loss.backward()
        opt2.step()
    finally:
        os.remove(ckpt_path)
        os.rmdir(tmpdir)
