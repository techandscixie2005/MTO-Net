# MTO-Net — Minimal Viable Loop

Minimal Tensor Orbital (MTO) network — a valence-adaptive latent orbital-like
module attached to the DetaNet equivariant GNN backbone.

## Minimal Loop Status

- [x] Git project scaffold
- [x] QM9S download script (Figshare API)
- [x] DetaNet backbone inspection
- [x] DetaNet adapter (exposes atom-level scalar features via `out_type='latent'`)
- [x] Minimal valence-adaptive MTO forward pass
- [x] Smoke test on synthetic molecules
- [x] One MTO atom-contribution map figure
- [x] Unit tests (valence, MTO shapes)
- [ ] QM9S dataset download (pending manual download)

## Quick Start

```bash
# Dependencies
module load miniforge3/25.11.0-1
pip install -r requirements.txt

# Smoke test (no DetaNet backbone needed)
python scripts/smoke_mto_forward.py

# Smoke test with DetaNet backbone
python scripts/smoke_mto_forward.py --use-detanet

# Generate one MTO map
python scripts/plot_one_mto_map.py --mol-index 0 --slot 0

# Inspect DetaNet backbone
python scripts/inspect_detanet.py

# Run tests
pytest tests/ -v
```

## Download QM9S

```bash
python scripts/download_qm9s.py --out data/qm9s
```

If the Figshare download fails, download manually from:
<https://figshare.com/articles/dataset/QM9S_dataset/24235333>

## Current Limitations

- Atom features are random/synthetic (no pretrained DetaNet weights loaded)
- MTO uses scalar features only (T tensor not yet used)
- K = total molecular valence electron count (no charge correction)
- No energy/force/Hessian/MD
- Single-slot visualization only
- QM9S download not yet tested
- MTO is NOT a physical molecular orbital
