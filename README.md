# Navier–Stokes-3D-Invariant-Zeta

**Proposed proof of global regularity for 3D incompressible Navier–Stokes via a new invariant ζ(t).**  
Reproducible code, documented tests, and benchmark scripts. MIT licensed.

---

## ✨ Core Contribution

We introduce the invariant
\[
\zeta(t)=\sqrt{\ \|u(t)\|_{H^{1/2}}^2\ +\ \beta\,\|\omega(t)\|_{L^2}^2\ }\!,
\]
coupling the **critical Sobolev norm** \(H^{1/2}\) with **vorticity intensity**.  
Our results and computations indicate that ζ(t) provides **global control** of the solution, excluding finite-time blow-up and yielding **global regularity** for 3D Navier–Stokes.

- Unified control: critical regularity + vorticity in one computable quantity.  
- Theory ⇄ numerics: invariant is implemented, monitored, and benchmarked in open code.  
- Reproducible science: scripts, tests, and notebooks to validate ζ(t) on 10 canonical flows.

---

## 📦 Repository Layout

Navier-Stokes-3D-Invariant-Zeta/ ├── 📁 src/ │   ├── 📄 init.py │   └── 📄 zeta_calculator.py         # 3D pseudo-spectral solver + ζ(t) + 10 ICs ├── 📁 scripts/ │   ├── 📄 run_case.py                # run a single case → data/.csv │   └── 📄 bench_all.py               # run all 10 cases → artifacts/run-/.csv ├── 📁 tests/ │   ├── 📄 test_zeta.py │   └── 📄 test_physical_identities.py ├── 📁 docs/ │   ├── 📄 index.md │   ├── 📄 invariant_zeta.md          # FR + EN theory pages │   └── 📄 howto_runs.md              # FR + EN run guides ├── 📁 data/                          # light results (csv)  [tracked: README.md, .gitkeep] ├── 📁 artifacts/                     # benchmark runs        [tracked: README.md, .gitkeep] ├── 📁 notebooks/ │   ├── 📄 README.md │   ├── 📄 analysis.ipynb │   └── 📄 compare_cases.ipynb ├── 📄 LICENSE (MIT) └── 📄 README.md (this file)

---

## 🚀 Quick Start

```bash
# (optional) create & activate a venv
# python -m venv .env && source .env/bin/activate

pip install -r requirements.txt

# Run a single case (CSV → data/)
python scripts/run_case.py --case taylor_green --N 32 --steps 120 --dt 0.004 --nu 0.01

# Run all 10 initial conditions (CSVs → artifacts/run-YYYYMMDD-HHMMSS/)
python scripts/bench_all.py --N 32 --steps 80 --dt 0.005 --nu 0.01

Open notebooks to visualize ζ(t) and energy:

jupyter lab
# open notebooks/analysis.ipynb or notebooks/compare_cases.ipynb


---

🧪 Tests

pytest -q

The suite checks: incompressibility preservation, non-negativity/finite diagnostics, short-step stability, and the availability of all 10 initial conditions.


---

📚 Documentation

Invariant ζ(t) (theory): see docs/invariant_zeta.md (FR + EN).

How to run (practical): see docs/howto_runs.md (FR + EN).

Notebooks for analysis: notebooks/analysis.ipynb, notebooks/compare_cases.ipynb.



---

🔬 Initial Conditions (10)

Beltrami (ABC), Taylor–Green, Lamb–Oseen, Compact bump, Rough (H^{1/2}), Axisymmetric swirl,
Truncated jet, Shear layer, Small-scales superposition, Double opposite vortices.


---

📜 License

MIT — see LICENSE.


---

✍️ Author

Mushagalusa Bashizi Grâce — Open, reproducible path toward a decisive invariant-based resolution of the 3D Navier–Stokes global regularity problem.


