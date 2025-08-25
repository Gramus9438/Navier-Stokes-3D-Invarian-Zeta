# Navier-Stokes-3D-Invariant-Zeta

**Global Regularity Proof for 3D Navier–Stokes via ζ(t) Invariant**  
Proposed solution to the **Millennium Prize Problem** on the global regularity of incompressible Navier–Stokes equations.  
Repository includes numerical validation code, data, reproduction scripts, and supporting documentation.  
Licensed under **MIT**.

---

# Navier–Stokes Global Regularity Proof via ζ(t) Invariant

This repository presents the official implementation and validation of a proposed **proof of global regularity** for the three–dimensional incompressible Navier–Stokes equations, addressing one of the **Clay Mathematics Institute’s Millennium Prize Problems**.

---

##  Core Innovation

We introduce a new **energy-type invariant ζ(t)** designed to control the nonlinear growth of solutions to the 3D Navier–Stokes system.  
This invariant encodes a dynamic balance between **kinetic energy** and **vorticity intensity**, providing a mechanism to prevent the formation of finite-time singularities.

---
‎
*"Contribution scientifique — Invariant ζ(t)"*
‎
‎ Novelty Statement
‎
‎The originality of this work lies not in the numerical solver itself — pseudo-spectral Navier–Stokes solvers are well-established — but in the introduction, definition, and validation of a new invariant:
‎
‎\zeta(t) \;=\; \sqrt{ \; \|u(t)\|_{H^{1/2}}^2 \;+\; \beta \,\|\omega(t)\|_{L^2}^2 \;}
‎
‎where
‎
‎ is the velocity field,
‎
‎ is the vorticity,
‎
‎ denotes the critical Sobolev norm,
‎
‎and  is a tunable constant.
‎
‎
‎
‎---
‎
‎Core novelty
‎
‎1. Unified invariant — Unlike classical approaches that study energy or enstrophy separately, ζ(t) couples the critical norm  with enstrophy in a single quantity.
‎
‎
‎2. Global control mechanism — The structure of ζ(t) suggests that if ζ(t) remains bounded, then finite-time singularities are excluded, implying global regularity of Navier–Stokes solutions.
‎
‎
‎3. Bridging theory and computation — The invariant is not only a theoretical construct, but also implemented numerically in a reproducible solver, allowing others to test and validate its robustness.
‎
‎
‎4. Critical space alignment — The use of the  Sobolev space aligns with known results on criticality for Navier–Stokes, but ζ(t) goes further by coupling it dynamically with vorticity intensity.
‎
‎
‎
‎
‎---
‎
‎Impact
‎
‎ζ(t) provides a new pathway to a rigorous proof of global regularity for 3D incompressible Navier–Stokes.
‎
‎It offers a computable diagnostic that can be monitored in numerical simulations, making the abstract theory concrete and testable.
‎
‎By releasing both the mathematical formulation and the open-source code, this repository sets a standard of reproducible mathematics and computational verification.

---

## 📂 Repository Content

- ✅ Core numerical solver with ζ(t) computation (`src/zeta_calculator.py`)  
- ✅ Validation on **10 canonical test flows** (Taylor–Green, Lamb–Oseen, Beltrami, Compact bump, Rough \(H^{1/2}\) data, etc.)  
- ✅ Pathological test cases (e.g., vortex filaments, shear layers, turbulent jets)  
- ✅ Reproduction scripts for numerical experiments (`scripts/`)  
- ✅ Documentation (`docs/`) including invariant definition and usage guidelines  
- ✅ Unit tests (`tests/`) ensuring reproducibility and physical consistency  

---

## 🔬 Numerical Validation Strategy

The validation methodology integrates:

- **Pseudo-spectral solver** with 3D FFT in periodic domains  
- **Dealiasing (2/3 rule)** for rigorous error control  
- **Comparison against analytical benchmarks** (Taylor–Green, Lamb–Oseen)  
- **Stress tests** on rough and near-singular initial data (Buckmaster–Vicol–type scenarios)  
- **Quantitative monitoring** of ζ(t), energy, enstrophy, helicity, and \(H^{1/2}\) norms  

---


##  Reference
This work accompanies the arXiv preprint: [link to be added]

## 👨‍💻 Author
Mushagalusa Bashizi Grâce
---

## ▶️ Quick Start

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
