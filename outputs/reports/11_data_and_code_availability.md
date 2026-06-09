# Data and Code Availability Statement

**Project:** MTO-Net: Valence-Adaptive Molecular Tensor Orbitals for Equivariant Molecular Response Learning
**Date:** 2026-06-09

## 1. Code Availability

The full source code for MTO-Net is available at:

- **GitHub:** https://github.com/techandscixie2005/MTO-Net
- **Branch:** `tensor-mto-v2`
- **Local path:** `/home/xiangyu_xie/MTO`
- **Commit hash:** aaccf1a (Stage B/C code completion)

The repository includes:
- Core MTO-Net model (`src/mto/`)
- Training and evaluation scripts (`scripts/`)
- Configuration files (`configs/`)
- Test suite (`tests/`, 75 passing)
- Analysis and plotting code (`scripts/eval/`, `scripts/figures/`)
- Slurm job templates (`jobs/`)

### Software Dependencies

```
python>=3.10
torch>=2.7
torch_geometric
e3nn>=0.4.4
rdkit
numpy
scipy
pandas
matplotlib
torch_cluster (for torch_geometric radius_graph)
```

Environment export: `outputs/reports/conda_env_export.yml` (on HPC server)

## 2. Data Availability

### QM9S Dataset

The QM9S dataset used in this study is available from Figshare at:
https://springernature.figshare.com/articles/dataset/QM9S/28314208

Original reference: DetaNet paper (arXiv:2406.04669)

**Download command:**
```python
python scripts/download_qm9s.py
```

**Files:**
- `qm9s.pt` (2.7 GB) — Main dataset: 129,817 molecules with dipole and polarizability labels
- `ir_boraden.csv` (8.2 GB) — Broadened IR spectra (3501 bins, 500-4000 cm⁻¹)
- `raman_boraden.csv` (8.2 GB) — Broadened Raman spectra (3501 bins, 500-4000 cm⁻¹)
- `uv_boraden.csv` (1.5 GB) — Broadened UV spectra (701 bins, 1.0-15.0 eV)

**Available labels in qm9s.pt:**
- Dipole moment (mu): [1, 3] vector
- Polarizability tensor (alpha): [1, 3, 3] matrix
- Atomic numbers (z), 3D coordinates (pos), SMILES strings

**Not available (explicitly absent from .pt):**
- Hessian, vibrational frequencies, normal modes
- Dipole derivatives, polarizability derivatives
- IR/Raman intensities at mode level
- UV transition energies, oscillator strengths
- NMR shielding constants

### Data Split

- Train/val/test split locked in: `outputs/splits/qm9s_split_stage_a.json`
- Split hash: `outputs/splits/qm9s_split_stage_a_hash.txt`
- All seeds use identical split for reproducibility

### Training Artifacts

Checkpoints, metrics, and analysis outputs are available at:
- `outputs/checkpoints/` (on HPC server)
- `outputs/metrics/`
- `outputs/analysis/`
- `outputs/figures/`

## 3. Reproducibility

### Key Configuration

| Parameter | Value |
|-----------|-------|
| Model parameters | 1,523,793 |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-5) |
| Batch size | 64 |
| Epochs | 50 (Stage A) |
| Seeds | 0, 1, 2, 3, 4 |
| MTO K mode | total_valence_electrons (K = N_val) |
| Activity gate | simple (sigmoid) |
| Diversity weight | 1e-3 |
| Split ratio | 5000 train / 2000 val / 2000 test |

### To Reproduce

```bash
# Local smoke test
python scripts/train_stage.py --stage a --config configs/train/stage_a.yaml --epochs 2 --seed 0

# Full Stage A on HPC (requires GPU + Slurm)
sbatch jobs/full_stage_a_multiseed.slurm
```

## 4. FAIR Compliance

| Principle | Status |
|-----------|--------|
| **Findable** | GitHub repo, Figshare dataset DOI |
| **Accessible** | Open source (MIT license), open data |
| **Interoperable** | Python/PyTorch ecosystem, CSV + PyG formats |
| **Reusable** | Configs, scripts, and documentation provided |

## 5. License

- MTO-Net code: MIT License
- DetaNet (third_party): MIT License
- QM9S dataset: CC BY 4.0 (Figshare)

## 6. Contact

- Repository: https://github.com/techandscixie2005/MTO-Net
- Issues and questions via GitHub Issues
