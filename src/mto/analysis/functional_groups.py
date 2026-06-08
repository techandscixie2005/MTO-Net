"""Functional group analysis using RDKit SMARTS patterns."""
import torch
import numpy as np


FUNCTIONAL_GROUPS = {
    "carbonyl": "[CX3]=[OX1]",
    "aldehyde": "[CX3H1](=O)[#6,#1]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxyl": "C(=O)[OX2H1,OX1-]",
    "hydroxyl": "[OX2H]",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "aromatic_ring": "a1aaaaa1",
    "nitrile": "C#N",
    "ether": "[OD2]([#6])[#6]",
    "fluoro": "[F]",
    "alkene": "C=C",
    "alkyne": "C#C",
    "amide": "[NX3][CX3](=[OX1])",
}


def detect_functional_groups(smiles_list):
    """Detect functional groups from SMILES strings using RDKit.

    Returns:
        dict[mol_idx] -> {"group_name": [atom_indices, ...]}
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except ImportError:
        return {}

    results = {}
    for mol_idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol_groups = {}
        for group_name, smarts in FUNCTIONAL_GROUPS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                mol_groups[group_name] = [list(m) for m in matches]
        if mol_groups:
            results[mol_idx] = mol_groups
    return results


def functional_group_enrichment(atom_contributions, group_atoms, all_atoms=None):
    """Compute enrichment: mean(contrib in group) / mean(contrib overall).

    Args:
        atom_contributions: dict mol_idx -> np.array[N_atoms] atom-level importance
        group_atoms: dict mol_idx -> dict group_name -> list[atom_indices]
        all_atoms: optional dict mol_idx -> int total atoms

    Returns:
        list of dicts with enrichment scores
    """
    records = []
    for mol_idx in atom_contributions:
        w = np.asarray(atom_contributions[mol_idx])
        w_mean = float(np.mean(w)) if len(w) > 0 else 1e-8
        for group_name, atom_lists in group_atoms.get(mol_idx, {}).items():
            all_group_atoms = [a for matches in atom_lists for a in matches]
            if not all_group_atoms:
                continue
            group_w = np.mean([w[a] for a in all_group_atoms if a < len(w)])
            enrichment = group_w / max(w_mean, 1e-8)
            records.append({
                "mol_idx": int(mol_idx),
                "group": group_name,
                "num_group_atoms": len(all_group_atoms),
                "enrichment": float(enrichment),
                "group_mean": float(group_w),
                "overall_mean": float(w_mean),
            })
    return records


def random_baseline_enrichment(atom_contributions, group_atoms, n_random=100, seed=0):
    """Compute random baseline enrichment by shuffling atom maps."""
    rng = np.random.RandomState(seed)
    random_scores = []
    for mol_idx in atom_contributions:
        w = np.asarray(atom_contributions[mol_idx])
        for _ in range(min(n_random, 20)):
            w_shuffled = rng.permutation(w)
            w_mean = float(np.mean(w_shuffled))
            for group_name, atom_lists in group_atoms.get(mol_idx, {}).items():
                all_group_atoms = [a for matches in atom_lists for a in matches]
                if not all_group_atoms:
                    continue
                group_w = np.mean([w_shuffled[a] for a in all_group_atoms if a < len(w_shuffled)])
                random_scores.append(group_w / max(w_mean, 1e-8))
    return {
        "mean": float(np.mean(random_scores)),
        "std": float(np.std(random_scores)),
        "n": len(random_scores),
    } if random_scores else {"mean": 1.0, "std": 0.0, "n": 0}


def compute_p_value(enrichment, random_stats):
    """Approximate p-value from random baseline distribution."""
    z = (enrichment - random_stats["mean"]) / max(random_stats["std"], 1e-8)
    # Normal approximation
    from math import erfc, sqrt
    p = float(erfc(abs(z) / sqrt(2)))
    return min(p, 1.0)
