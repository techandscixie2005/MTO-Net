#!/usr/bin/env python3
"""Generate figures 16-18 from real experiment metrics data."""
import json, os, sys, numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
OUT_DIR = os.path.join(PROJ_ROOT, "outputs", "figures", "final")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
                     "figure.dpi": 150, "savefig.bbox": "tight", "savefig.dpi": 150})


def load_json(*parts):
    path = os.path.join(PROJ_ROOT, *parts)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def make_fig16():
    """Baseline ablation summary — normalized test MAE for 5 methods × 3 seeds."""
    data = load_json("outputs", "metrics", "baselines", "baseline_comparison.json")
    if data is None:
        print("SKIP fig16: no baseline data")
        return

    methods = ["full_mto", "no_sign_mto", "fixed_k_mto", "direct_readout", "attention_pooling"]
    labels = ["Full\nMTO", "No\nSign", "Fixed\nK", "Direct\nReadout", "Attention\nPool"]

    seeds = sorted(set(d["seed"] for d in data))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, task, ylabel in [(axes[0], "mu", "Normalized mu MAE"),
                               (axes[1], "alpha", "Normalized alpha MAE")]:
        x = np.arange(len(methods))
        width = 0.25
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for si, seed in enumerate(seeds):
            seed_data = [d for d in data if d["seed"] == seed]
            vals = []
            for m in methods:
                match = [d for d in seed_data if d["method"] == m]
                vals.append(match[0]["test_norm"][task] if match else np.nan)
            bars = ax.bar(x + (si - 1) * width, vals, width, label=f"seed {seed}",
                          color=colors[si], edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{task.upper()} Prediction (normalized MAE)")
        ax.legend(fontsize=7, loc="upper right" if task == "mu" else "upper left")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Fig 16: Baseline & Ablation — Normalized Test MAE (5k mols, 20 ep, 3 seeds)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for fmt in ["pdf", "png"]:
        path = os.path.join(OUT_DIR, f"fig16_baseline_ablation_summary.{fmt}")
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


def make_fig17():
    """Stage transfer stability — raw + normalized metrics + S_sub."""
    data = load_json("outputs", "stage_transfer", "stage_transfer_results.json")
    if data is None:
        print("SKIP fig17: no stage transfer data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: Raw MAE by condition (mu task)
    ax = axes[0]
    conditions = ["mu_only", "alpha_only", "mu_alpha"]
    x_labels = ["mu-only", "alpha-only", "mu+alpha"]
    xs = np.arange(len(conditions))
    width = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for si, entry in enumerate(data):
        mu_raw = []
        for cn in conditions:
            v = entry["conditions"][cn]["test_raw"].get("mu", np.nan)
            mu_raw.append(v if v is not None and not np.isnan(v) else 0)
        ax.bar(xs + (si - 1) * width, mu_raw, width, label=f"seed {entry['seed']}",
               color=colors[si], edgecolor="white", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Raw mu MAE (Debye)")
    ax.set_title("Mu prediction by training condition")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: Normalized MAE by condition
    ax = axes[1]
    for si, entry in enumerate(data):
        mu_norm = []
        for cn in conditions:
            v = entry["conditions"][cn]["test_norm"].get("mu", np.nan)
            mu_norm.append(v if v is not None and not np.isnan(v) else 0)
        ax.bar(xs + (si - 1) * width, mu_norm, width, label=f"seed {entry['seed']}",
               color=colors[si], edgecolor="white", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Normalized mu MAE")
    ax.set_title("Mu prediction (normalized)")
    ax.grid(axis="y", alpha=0.3)

    # Panel C: S_sub heatmap
    ax = axes[2]
    seeds_list = [d["seed"] for d in data]
    s_sub_vals = [d["subspace_sim"]["mu_only_vs_alpha_only"] for d in data]
    ax.bar(range(len(seeds_list)), s_sub_vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_xticks(range(len(seeds_list)))
    ax.set_xticklabels([f"Seed {s}" for s in seeds_list], fontsize=9)
    ax.set_ylabel("S_sub (mu-only vs alpha-only)")
    ax.set_title("MTO Subspace Transfer")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="S_sub = 0.5")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Fig 17: Stage Transfer Stability (Intra-Stage-A, 5k mols, 20 ep)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for fmt in ["pdf", "png"]:
        path = os.path.join(OUT_DIR, f"fig17_stage_transfer_stability.{fmt}")
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


def make_fig18():
    """Frozen probe — comparison of 3 conditions."""
    import glob as _glob
    # Try HPC path first, then local
    fp_path = None
    for base in [os.path.join(PROJ_ROOT, "outputs", "frozen_probe"),
                 "/data/home/scwc008/run/xxy/MTO/outputs/frozen_probe"]:
        candidate = os.path.join(base, "frozen_probe_results.json")
        if os.path.exists(candidate):
            fp_path = candidate
            break

    if fp_path is None:
        print("SKIP fig18: no frozen_probe_results.json found locally or on HPC")
        # Check if HPC data available
        return

    data = load_json(*fp_path.split(os.sep)[-3:]) if "MTO" in fp_path else None
    if data is None:
        with open(fp_path) as f:
            data = json.load(f)
    if data is None:
        print("SKIP fig18: could not load data")
        return

    conditions = ["frozen_mto", "frozen_direct", "from_scratch"]
    c_labels = ["Frozen MTO\n+ new readout", "Frozen DetaNet\n+ new readout", "From Scratch\n(full training)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, task in [(axes[0], "mu"), (axes[1], "alpha")]:
        xs = np.arange(len(conditions))
        width = 0.25
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for si, entry in enumerate(data):
            vals = []
            for cn in conditions:
                if cn in entry:
                    v = entry[cn]["test"].get(task, np.nan)
                    vals.append(v if v is not None and not np.isnan(v) else 0)
                else:
                    vals.append(np.nan)
            ax.bar(xs + (si - 1) * width, vals, width, label=f"seed {entry['seed']}",
                   color=colors[si], edgecolor="white", linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(c_labels, fontsize=7)
        ax.set_ylabel(f"Raw {task} MAE")
        ax.set_title(f"{task.upper()} Prediction")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Fig 18: Frozen Probe — Representation Reusability (5k mols, 20 ep)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for fmt in ["pdf", "png"]:
        path = os.path.join(OUT_DIR, f"fig18_frozen_probe_reuse.{fmt}")
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


def main():
    print("Generating Figure 16...")
    make_fig16()
    print("Generating Figure 17...")
    make_fig17()
    print("Generating Figure 18...")
    make_fig18()
    print("Done.")


if __name__ == "__main__":
    main()
