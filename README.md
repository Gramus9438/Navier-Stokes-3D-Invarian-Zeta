# Navier-Stokes-3D-Invariant-Zeta

**Global Regularity Proof for 3D Navier–Stokes via ζ(t) Invariant**  
Proposed solution to the **Millennium Prize Problem** on the global regularity of incompressible Navier–Stokes equations.  
Repository includes numerical validation code, data, reproduction scripts, and supporting documentation.  
Licensed under **MIT**.

---

# Navier–Stokes Global Regularity Proof via ζ(t) Invariant

This repository presents the official implementation and validation of a proposed **proof of global regularity** for the three–dimensional incompressible Navier–Stokes equations, addressing one of the **Clay Mathematics Institute’s Millennium Prize Problems**.

---

## 🌟 Core Innovation

We introduce a new **energy-type invariant ζ(t)** designed to control the nonlinear growth of solutions to the 3D Navier–Stokes system.  
This invariant encodes a dynamic balance between **kinetic energy** and **vorticity intensity**, providing a mechanism to prevent the formation of finite-time singularities.

---

## 📐 Mathematical Definition

The invariant ζ(t) is defined as:

$$
\zeta(t) = \int_{\mathbb{R}^3} \left( 
\frac{|\mathbf{u}(x,t)|^3}{1 + |\mathbf{u}(x,t)|} \;+\; 
|\nabla \times \mathbf{u}(x,t)|^{3/2} 
\right) e^{-\|x\|} \, dx
$$

where  
- **u(x,t)** is the velocity field,  
- **∇×u** is the vorticity field,  
- and the exponential weight ensures integrability on \(\mathbb{R}^3\).

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
