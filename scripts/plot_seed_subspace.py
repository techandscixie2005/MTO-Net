#!/usr/bin/env python3
"""Plot seed subspace stability from saved metrics."""
import argparse, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="outputs/metrics/seed_subspace_stability.json")
    parser.add_argument("--out-dir", default="outputs/figures/stability")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.metrics):
        print(f"No metrics found at {args.metrics}, generating demo plot")
        seeds = ["seed_0", "seed_1", "seed_2"]
        n = len(seeds)
        sims = np.random.uniform(0.3, 0.9, n * (n - 1) // 2)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        data = [{"seed_a": seeds[i], "seed_b": seeds[j], "similarity": float(s)}
                for (i, j), s in zip(pairs, sims)]
    else:
        with open(args.metrics) as f:
            result = json.load(f)
        data = result.get("pairs", [])

    if not data:
        print("No data to plot")
        return

    # Boxplot
    similarities = [d["similarity"] for d in data]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.boxplot(similarities)
    ax1.set_title("Seed Subspace Similarity")
    ax1.set_ylabel("Subspace similarity (S_sub)")
    ax1.set_xticklabels(["Seed pairs"])

    # Heatmap
    seeds = sorted(set(d["seed_a"] for d in data) | set(d["seed_b"] for d in data))
    n = len(seeds)
    mat = np.eye(n)
    for d in data:
        i = seeds.index(d["seed_a"])
        j = seeds.index(d["seed_b"])
        mat[i, j] = d["similarity"]
        mat[j, i] = d["similarity"]

    im = ax2.imshow(mat, vmin=0, vmax=1, cmap="RdYlBu_r")
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(seeds, rotation=45)
    ax2.set_yticklabels(seeds)
    ax2.set_title("Seed Pair Similarity")
    plt.colorbar(im, ax=ax2, label="S_sub")

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "seed_pair_heatmap.png")
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved: {fig_path}")

if __name__ == "__main__":
    main()
