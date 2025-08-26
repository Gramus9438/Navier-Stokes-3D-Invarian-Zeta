# scripts/bench_all.py
"""
Run all 10 initial conditions and save results in artifacts/run-<timestamp>/.

Usage:
    python scripts/bench_all.py --N 32 --steps 100 --dt 0.01 --nu 0.01
"""

import argparse
import os
import time
import csv
from src.zeta_calculator import run_simulation, NSConfig, IC_LIBRARY

def main():
    parser = argparse.ArgumentParser(description="Benchmark all invariant ζ(t) test cases")
    parser.add_argument("--N", type=int, default=32, help="Grid size per dimension")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument("--nu", type=float, default=0.01, help="Viscosity")
    parser.add_argument("--save_every", type=int, default=10, help="Save every N steps")
    parser.add_argument("--out", type=str, default="artifacts", help="Output directory")

    args = parser.parse_args()
    cfg = NSConfig(N=args.N, steps=args.steps, dt=args.dt, nu=args.nu, save_every=args.save_every)

    # timestamped folder
    run_id = time.strftime("run-%Y%m%d-%H%M%S")
    outdir = os.path.join(args.out, run_id)
    os.makedirs(outdir, exist_ok=True)

    print(f"Running benchmark of {len(IC_LIBRARY)} cases...")
    for case in IC_LIBRARY.keys():
        print(f"  -> {case}")
        out = run_simulation(case, ns_cfg=cfg)

        outfile = os.path.join(outdir, f"{case}.csv")
        with open(outfile, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "energy", "enstrophy", "helicity", "H12", "zeta"])
            for i in range(len(out["t"])):
                writer.writerow([out["t"][i], out["energy"][i], out["enstrophy"][i], out["helicity"][i], out["H12"][i], out["zeta"][i]])

    print(f"All results stored in {outdir}")

if __name__ == "__main__":
    main()
