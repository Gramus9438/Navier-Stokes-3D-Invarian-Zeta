# scripts/run_case.py
"""
Run a single Navier–Stokes test case with ζ(t) monitoring.

Usage:
    python scripts/run_case.py --case taylor_green --N 32 --steps 100 --dt 0.01 --nu 0.01
"""

import argparse
import numpy as np
import os
from src.zeta_calculator import run_simulation, NSConfig

def main():
    parser = argparse.ArgumentParser(description="Run a Navier–Stokes case with invariant ζ(t)")
    parser.add_argument("--case", type=str, required=True, help="Case name (see IC_LIBRARY in zeta_calculator.py)")
    parser.add_argument("--N", type=int, default=32, help="Grid size per dimension")
    parser.add_argument("--steps", type=int, default=100, help="Number of time steps")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step size")
    parser.add_argument("--nu", type=float, default=0.01, help="Viscosity")
    parser.add_argument("--save_every", type=int, default=10, help="Save every N steps")
    parser.add_argument("--out", type=str, default="data", help="Output directory")

    args = parser.parse_args()

    # configure solver
    cfg = NSConfig(N=args.N, steps=args.steps, dt=args.dt, nu=args.nu, save_every=args.save_every)

    print(f"Running case '{args.case}' with N={args.N}, steps={args.steps}, dt={args.dt}, nu={args.nu}")
    out = run_simulation(args.case, ns_cfg=cfg)

    # prepare output dir
    os.makedirs(args.out, exist_ok=True)
    outfile = os.path.join(args.out, f"{args.case}_N{args.N}_nu{args.nu}.csv")

    # save results
    import csv
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "energy", "enstrophy", "helicity", "H12", "zeta"])
        for i in range(len(out["t"])):
            writer.writerow([out["t"][i], out["energy"][i], out["enstrophy"][i], out["helicity"][i], out["H12"][i], out["zeta"][i]])

    print(f"Results saved to {outfile}")

if __name__ == "__main__":
    main()
