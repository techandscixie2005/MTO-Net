#!/usr/bin/env python3
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patterns", nargs="+",
                       default=["carbonyl", "aromatic", "amine", "nitro", "nitrile"])
    parser.add_argument("--out-dir", default="outputs/figures/functional_groups")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("Functional group analysis (checkpoint: " + args.checkpoint + ")")
    print("Patterns: " + str(args.patterns))
    print("Note: Full functional group detection requires RDKit.")

    for pattern in args.patterns:
        fig, ax = plt.subplots(figsize=(6, 4))
        msg = "Functional group: " + pattern + "\n\n(Requires trained model\nand RDKit)"
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=12)
        ax.set_title("MTO maps for " + pattern)
        path = os.path.join(args.out_dir, pattern + "_mto.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print("  Placeholder: " + path)

if __name__ == "__main__":
    main()
