# scripts/plot_zeta.py
"""
Plot ζ(t) (and optionally Energy) from a results CSV.

Usage examples:
    python scripts/plot_zeta.py --in data/taylor_green_N32_nu0.01.csv
    python scripts/plot_zeta.py --in data/taylor_green_demo.csv --y zeta energy --out figures/taylor_green.png
    python scripts/plot_zeta.py --in data/taylor_green_demo.csv --title "Taylor–Green (demo)" --dpi 150

Input CSV is expected to have columns:
    t, energy, enstrophy, helicity, H12, zeta
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description="Plot ζ(t) and Energy from a CSV file.")
    p.add_argument("--in", dest="inp", required=True, help="Path to input CSV")
    p.add_argument("--out", dest="out", default=None, help="Path to output PNG (default: <input>_plot.png)")
    p.add_argument("--y", nargs="+", default=["zeta"], choices=["zeta", "energy"],
                   help="Series to plot (default: zeta). You may pass both: --y zeta energy")
    p.add_argument("--title", default=None, help="Figure title (optional)")
    p.add_argument("--dpi", type=int, default=120, help="PNG resolution (default: 120)")
    p.add_argument("--show", action="store_true", help="Display the figure interactively")

    args = p.parse_args()
    csv_path = Path(args.inp)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Output file
    if args.out is None:
        out_path = csv_path.with_suffix("").as_posix() + "_plot.png"
    else:
        out_path = args.out
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path)
    for col in ["t"] + args.y:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {csv_path}. Columns found: {list(df.columns)}")

    # Plot
    plt.figure()
    for col in args.y:
        plt.plot(df["t"], df[col], label=col)

    plt.xlabel("t")
    # If single series, label y-axis accordingly
    if len(args.y) == 1:
        plt.ylabel(args.y[0])
    else:
        plt.ylabel("value")

    if args.title:
        plt.title(args.title)
    else:
        plt.title(f"Plot from {csv_path.name}")

    plt.legend()
    plt.tight_layout()

    # Save and/or show
    plt.savefig(out_path, dpi=args.dpi)
    print(f"Saved figure to: {out_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
