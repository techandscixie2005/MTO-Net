import torch

VALENCE_ELECTRONS = {
    1: 1,   # H
    2: 2,   # He (inert but needed for completeness)
    3: 1,   # Li
    4: 2,   # Be
    5: 3,   # B
    6: 4,   # C
    7: 5,   # N
    8: 6,   # O
    9: 7,   # F
}


def molecular_valence_electrons(z: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    max_z = max(int(z.max().item()), max(VALENCE_ELECTRONS.keys()))
    valence_map = torch.zeros(max_z + 1, dtype=torch.long, device=z.device)
    for atomic_num, valence in VALENCE_ELECTRONS.items():
        valence_map[atomic_num] = valence

    unknown = (valence_map[z] == 0) & (z > 0)
    if unknown.any():
        bad_z = z[unknown].unique().tolist()
        raise ValueError(
            f"Unknown atomic numbers {bad_z} "
            f"- add valence electron count to VALENCE_ELECTRONS"
        )

    atom_valence = valence_map[z]
    K_per_mol = torch.zeros(batch.max().item() + 1, dtype=torch.long, device=z.device)
    K_per_mol.scatter_add_(0, batch, atom_valence)
    return K_per_mol
