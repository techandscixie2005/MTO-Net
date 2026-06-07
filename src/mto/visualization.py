import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATOM_COLORS = {
    1: "#CCCCCC",
    6: "#333333",
    7: "#3050F8",
    8: "#FF2010",
    9: "#90E050",
}
ATOM_LABELS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def plot_mto_map(pos, z, coeff, mol_label="unknown", slot_idx=0,
                 K=None, is_synthetic=False,
                 save_path="outputs/figures/one_mto_map.png"):
    pos_np = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.array(pos)
    z_np = z.detach().cpu().numpy() if hasattr(z, "detach") else np.array(z)
    coeff_np = coeff.detach().cpu().numpy() if hasattr(coeff, "detach") else np.array(coeff)
    coeff_np = np.asarray(coeff_np).flatten()

    signs = np.sign(coeff_np)

    fig = plt.figure(figsize=(14, 5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for i, (x, y, zz) in enumerate(pos_np):
        an = int(z_np[i])
        color = ATOM_COLORS.get(an, "#AAAAAA")
        label = ATOM_LABELS.get(an, "?")
        size = 100 + 500 * abs(float(coeff_np[i]))
        ax.scatter(x, y, zz, c=color, s=size, edgecolors="black",
                   linewidth=0.5, alpha=0.9)
        ax.text(x, y, zz, f"  {label}", fontsize=8, color="black")

    ax.set_xlabel("x (A)")
    ax.set_ylabel("y (A)")
    ax.set_zlabel("z (A)")
    ax.set_title(f"3D Structure {mol_label}")

    ax2 = fig.add_subplot(1, 2, 2)
    N = len(coeff_np)
    colors_bar = ["#2E86AB" if s > 0 else "#D64045" for s in signs]
    ax2.bar(range(N), coeff_np, color=colors_bar, edgecolor="black", linewidth=0.5)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xticks(range(N))
    labels_x = [f"{ATOM_LABELS.get(int(zi), '?')}({i})"
                for i, zi in enumerate(z_np)]
    ax2.set_xticklabels(labels_x, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("c_ki (atom contribution)")
    ax2.set_xlabel("Atom index")
    ax2.set_title(f"Slot {slot_idx} Contributions")

    synth_tag = " [SYNTHETIC]" if is_synthetic else ""
    k_str = f"K={K}" if K is not None else ""
    fig.suptitle(
        f"MTO Atom-Contribution Map - {mol_label}{synth_tag}  "
        f"slot={slot_idx}  {k_str}",
        fontsize=12, fontweight="bold")
    plt.tight_layout()

    for fmt, ext in [("png", ".png"), ("pdf", ".pdf")]:
        sp = save_path.replace(".png", ext)
        fig.savefig(sp, dpi=150, bbox_inches="tight")
        print(f"Saved: {sp}")

    plt.close(fig)
    return save_path


def plot_mto_summary(all_coeffs, all_labels,
                     save_path="outputs/figures/mto_summary.png"):
    n = len(all_coeffs)
    cols = min(3, n)
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for idx, (coeff, label) in enumerate(zip(all_coeffs, all_labels)):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        abs_c = np.abs(np.asarray(coeff).flatten())
        ax.bar(range(len(abs_c)), abs_c, color="#2E86AB",
               edgecolor="black", linewidth=0.5)
        ax.set_title(label, fontsize=8)
        ax.set_ylabel("|c_ki|")
        ax.set_xlabel("Atom idx")

    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].set_visible(False)

    fig.suptitle("MTO Atom Contribution Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path
