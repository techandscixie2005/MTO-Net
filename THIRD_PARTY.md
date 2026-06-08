# Third-Party Dependencies

## DetaNet

- **Source:** https://github.com/techandscixie2005/DetaNet
- **Commit:** `c94892c` (vendor snapshot in `third_party/DetaNet/`)
- **License:** MIT
- **Vendor method:** Direct file copy (not git submodule)
- **Modifications:** None
- **Compatibility patches:** `src/mto/compat.py` applies runtime monkey-patches for:
  - `torch_geometric.nn.radius_graph` → `torch_cluster.radius_graph` (pyg-lib 0.6+ unavailable for torch 2.7)
  - `torch.serialization.add_safe_globals([slice])` (e3nn 0.4.4 + torch 2.7)
  - `DetaNet(scale=None)` (avoids tuple*float with `out_type=latent`)

No DetaNet source files have been modified. All patches are runtime-only.

## QM9S Dataset

- **Source:** https://figshare.com/articles/dataset/QM9S_dataset/24235333
- **Reference:** Zou et al. (2023), "A deep learning model for predicting selected organic molecular spectra", Nature Computational Science, https://doi.org/10.1038/s43588-023-00550-y
- **License:** CC BY 4.0 (Figshare default)
- **Usage:** Download for non-commercial research

## Key Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.7.0+cu128 | Deep learning framework |
| e3nn | 0.4.4 | E(3)-equivariant neural networks |
| torch_geometric | 2.8.0 | Graph neural networks |
| torch_cluster | 1.6.3 | Graph radius graph operations |
| torch_scatter | 2.1.2 | Scatter operations |
| torch_sparse | 0.6.18 | Sparse matrix operations |
| torch_spline_conv | 1.2.2 | Spline-based convolutions |
| numpy | 2.2.6 | Numerical computing |
| matplotlib | 3.10.9 | Visualization |
