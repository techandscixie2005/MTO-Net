# MTO-Net: Molecular Tensor Orbital Network

**Equivariant deep learning for molecular electronic-response modes atop DetaNet**

## Project Goal

MTO-Net asks: *Can atom-centered equivariant molecular representations be reorganized into stable, reusable, chemically meaningful molecule-level latent electronic-response modes?*

MTO (Molecular Tensor Orbital) is **not** a physical molecular orbital, not a Hamiltonian eigenstate, and not an explicit wavefunction. It is an **orbital-like latent molecular mode** in the DetaNet / equivariant representation space — a learned reorganization of atom-level features into interpretable, stable molecule-level channels.

## Architecture

```
Molecule (z, pos)
        │
        ▼
┌─────────────────────┐
│   DetaNet Backbone  │  ← equivariant message passing (unchanged)
│    (third_party/)   │
└─────────┬───────────┘
          │ atom features S [N_atoms, C]
          ▼
┌─────────────────────┐
│ ValenceAdaptiveMTO  │  ← signed routing with valence-adaptive K
│                     │    K = total valence electrons per molecule
│                     │    a_ki = softmax(route_mlp([h_i, atom_emb, slot_emb]))
│                     │    s_ki = tanh(sign_mlp(...))
│                     │    c_ki = a_ki * s_ki / sum(|c_ki|)
│                     │    O_k = sum_i c_ki * h_i
└─────────┬───────────┘
          │ MTO slots O [B, K_max, C]
          ▼
┌─────────────────────┐
│  MultiHead Readout  │  ← one head per property
│   mu, alpha, IR,    │
│   Raman, UV         │
└─────────────────────┘
```

## Relationship to DetaNet

MTO-Net uses DetaNet as its **atom-level equivariant tensor backbone**. The input-to-atom-representation path inside DetaNet is preserved unchanged; MTO is added only **after** the updated atom-level representation.

- DetaNet: `z, pos → S [N, C], T [N, vdim]` (scalar + irrep tensor)
- MTO: `S → O [B, K, C]` (valence-adaptive slot features)
- Readout: `O → mu, alpha, IR, Raman, UV`

**DetaNet source:** https://github.com/techandscixie2005/DetaNet  
**DetaNet commit:** c94892c (vendor snapshot)

## QM9S Dataset

**Source:** https://figshare.com/articles/dataset/QM9S_dataset/24235333  
**Reference:** Zou et al. (2023), Nature Computational Science, https://doi.org/10.1038/s43588-023-00550-y

QM9S contains 130k organic molecules with:
- Dipole moment vector (mu)
- Polarizability tensor (alpha)
- IR spectrum (500-4000 cm⁻¹)
- Raman spectrum (500-4000 cm⁻¹)
- UV-Vis spectrum (1.5-13.5 eV)
- Quadrupole, octupole moments, HOMO-LUMO gap, excitation energies

### Download

```bash
python scripts/download_qm9s.py --out data/qm9s
```

If the HPC cannot reach Figshare, download locally and transfer:

```bash
# Locally
curl -L -o qm9s.pt "https://ndownloader.figshare.com/files/42544564"
scp qm9s.pt bjhpc_xxy_1:/data/home/scwc008/run/xxy/MTO/data/qm9s/
```

## Pipeline

### Stages

| Stage | Properties | Description |
|-------|-----------|-------------|
| A | mu, alpha | Dipole + polarizability |
| B | mu, alpha, IR, Raman | + vibrational spectra |
| C | mu, alpha, IR, Raman, UV | + electronic spectra |

### Smoke Test (32 molecules)

```bash
python scripts/make_qm9s_subset.py --name smoke --num-mols 32 --seed 0
python scripts/prepare_qm9s.py --data-dir data/qm9s/subset_smoke
python scripts/train_stage.py --stage stage_a --data-dir data/qm9s/subset_smoke/processed --epochs 10 --seed 0
python scripts/eval_model.py --checkpoint outputs/checkpoints/stage_a_seed0/best.pt --data-dir data/qm9s/subset_smoke/processed
python scripts/plot_mto_maps.py --checkpoint outputs/checkpoints/stage_a_seed0/best.pt --num-mols 4 --top-slots 3
```

### Medium Run (512 molecules)

```bash
python scripts/run_all_medium.py
```

### Full QM9S Run (Slurm)

```bash
sbatch jobs/full_stage_a_multiseed.slurm
sbatch jobs/full_stage_b_transfer.slurm
sbatch jobs/full_stage_c_uv.slurm
```

## Stability Analysis

MTO slots have permutation/sign/gauge freedom. We use **subspace similarity**:

```
S_sub = trace(Q_a Q_a^T Q_b Q_b^T) / r
```

where Q_a, Q_b are QR-orthogonalized bases from top-r MTO contributions.

### Analyses

- **Seed stability**: pairwise subspace similarity across training seeds
- **Stage stability**: correlation of atom contribution maps across stages
- **Slot intervention**: mask one MTO slot, measure prediction delta

## Visualization

```bash
python scripts/plot_mto_maps.py --checkpoint <ckpt> --num-mols 32 --top-slots 5
python scripts/plot_seed_subspace.py --stage stage_a
python scripts/plot_stage_transfer.py --mols selected
python scripts/plot_functional_groups.py --checkpoint <ckpt> --patterns carbonyl aromatic amine
```

## Requirements

- Python 3.10+
- PyTorch 2.7+
- e3nn 0.4.4
- torch_geometric 2.8+
- torch_cluster, torch_scatter, torch_sparse, torch_spline_conv
- matplotlib, numpy

Environment: `conda activate py310-torch270-vllm090`

## Tests

```bash
pytest tests/ -v
```

## License

MIT
