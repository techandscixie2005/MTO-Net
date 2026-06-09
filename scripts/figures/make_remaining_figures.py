#!/usr/bin/env python3
"""Generate remaining figures from baseline comparison results.

fig16: baseline/ablation summary (bar chart comparison)
fig17: stage transfer stability (placeholder - needs full Stage B/C runs)
fig18: frozen probe reuse (placeholder - needs frozen probe experiment)
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "outputs/figures/final"
os.makedirs(OUT_DIR, exist_ok=True)

NATURE_COLORS = ["#2A4B7C", "#6B8E23", "#B8860B", "#8B0000", "#4A4A4A"]
METHOD_LABELS = {
    "full_mto": "Full MTO",
    "no_sign_mto": "No-sign MTO",
    "fixed_k_mto": "Fixed-K MTO",
    "direct_readout": "Direct Readout",
    "attention_pooling": "Attention Pooling",
}


def make_fig16(baseline_path="outputs/metrics/baselines/baseline_comparison.json"):
    """Bar chart: test MAE comparison across baselines."""
    print("Generating fig16: baseline comparison...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            results = json.load(f)

        names = [r["name"] for r in results]
        labels = [METHOD_LABELS.get(n, n) for n in names]
        mu_mae = [r["test"].get("mu", float("nan")) for r in results]
        alpha_mae = [r["test"].get("alpha", float("nan")) for r in results]
        params = [r["params"] for r in results]

        colors = NATURE_COLORS[:len(names)]

        # mu MAE
        bars1 = ax1.bar(range(len(labels)), mu_mae, color=colors, edgecolor="white", linewidth=0.5)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax1.set_ylabel("mu MAE (standardized)")
        ax1.set_title("Dipole Moment Prediction")
        for bar, val in zip(bars1, mu_mae):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        # alpha MAE
        bars2 = ax2.bar(range(len(labels)), alpha_mae, color=colors, edgecolor="white", linewidth=0.5)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax2.set_ylabel("alpha MAE (standardized)")
        ax2.set_title("Polarizability Prediction")
        for bar, val in zip(bars2, alpha_mae):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        fig.suptitle("Figure 16: Baseline and Ablation Comparison — Stage A (mu, alpha)",
                     fontsize=13, fontweight="bold")
    else:
        # Placeholder
        ax1.text(0.5, 0.5, f"Baseline results pending\n(Job on HPC)\nData: {baseline_path}",
                 ha="center", va="center", transform=ax1.transAxes, fontsize=12, color="gray")
        ax2.text(0.5, 0.5, "Run scripts/eval/run_baselines.py\nto generate comparison data",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=12, color="gray")
        fig.suptitle("Figure 16: Baseline Comparison — PENDING", fontsize=13, fontweight="bold")

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"fig16_baseline_ablation_summary.{fmt}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig16 done.")


def make_fig17():
    """Stage transfer stability — placeholder."""
    print("Generating fig17: stage transfer stability...")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5,
            "Figure 17: Stage Transfer Stability\n\n"
            "Requires full Stage B/C training runs.\n"
            "Stage B/C smoke tests passed (code verified).\n"
            "Full multi-stage training pending HPC GPU time.\n\n"
            "Expected analysis:\n"
            "- Stage A -> Stage B MTO map correlation\n"
            "- Stage B -> Stage C MTO map correlation\n"
            "- mu/alpha prediction retention after adding IR/Raman/UV\n"
            "- Subspace similarity across stages",
            ha="center", va="center", fontsize=12, color="#555",
            transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("Figure 17: Stage Transfer Stability — PENDING", fontsize=13, fontweight="bold")

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"fig17_stage_transfer_stability.{fmt}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig17 done.")


def make_fig18():
    """Frozen probe reuse — placeholder."""
    print("Generating fig18: frozen probe reuse...")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5,
            "Figure 18: Frozen Probe Reuse\n\n"
            "Requires frozen probe experiment:\n"
            "1. Train Stage A (mu + alpha) — COMPLETE\n"
            "2. Freeze backbone + MTO module\n"
            "3. Train only Stage B/C readout heads\n"
            "4. Compare vs from-scratch and full fine-tuning\n\n"
            "Code ready: src/mto/analysis/frozen_probe.py\n"
            "Config ready: configs/frozen_probe.yaml\n\n"
            "Experiment pending HPC GPU time.",
            ha="center", va="center", fontsize=12, color="#555",
            transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("Figure 18: Frozen Probe Reuse — PENDING", fontsize=13, fontweight="bold")

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"fig18_frozen_probe_reuse.{fmt}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig18 done.")


def make_spectral_figures(metrics_dir="outputs/metrics"):
    """Stage B/C spectral diagnostic figures if data available."""
    print("Generating spectral figures...")

    # Try loading Stage B smoke metrics
    stage_b_path = os.path.join(metrics_dir, "stage_b_seed100_metrics.json")
    stage_c_path = os.path.join(metrics_dir, "stage_c_seed101_metrics.json")

    if os.path.exists(stage_b_path):
        with open(stage_b_path) as f:
            b_metrics = json.load(f)
        print(f"  Stage B smoke: best_val_loss={b_metrics.get('best_val_loss', 'N/A')}")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5,
                f"Stage B Spectral Smoke Test Results\n\n"
                f"Job: 85917 (A800 GPU)\n"
                f"Tasks: mu, alpha, IR (3501 bins), Raman (3501 bins)\n"
                f"Best val loss: {b_metrics.get('best_val_loss', 'N/A'):.4f}\n\n"
                f"Code path verified. Full training pending.\n"
                f"Spectral CSV data available (ir_boraden.csv, raman_boraden.csv).",
                ha="center", va="center", fontsize=12, color="#2A4B7C",
                transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        for fmt in ["pdf", "png"]:
            fig.savefig(os.path.join(OUT_DIR, f"fig_supp_stage_b_spectral_smoke.{fmt}"),
                        dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Spectral supplementary fig done.")


if __name__ == "__main__":
    make_fig16()
    make_fig17()
    make_fig18()
    make_spectral_figures()
    print("\nAll remaining figures generated.")
