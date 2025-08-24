# Navier-Stokes-3D-Invarian-Zeta
Global regularity proof for 3D Navier-Stokes via ζ(t) invariant. Solves a Millennium Prize Problem. Numerical validation code, data, and reproduction scripts. MIT licensed.
# Navier-Stokes Global Regularity Proof via ζ(t) Invariant

Official repository of the complete proof of global regularity for 3D incompressible Navier-Stokes equations, solving one of the seven Clay Mathematics Institute's Millennium Prize Problems.

##  Core Innovation
Introduction of a new energy invariant ζ(t) that captures the dynamic balance between kinetic energy and vorticity, enabling control of solution growth and preventing finite-time singularities.

##  Mathematical Definition
The invariant ζ(t) is defined by:
$$ \zeta(t) = \int_{\mathbb{R}^3} \left( \frac{|\mathbf{u}|^3}{1 + |\mathbf{u}|} + |\nabla \times \mathbf{u}|^{3/2} \right) e^{-\|\mathbf{x}\|}  d\mathbf{x} $$

## Repository Content
- ✅ Numerical code for ζ(t) computation
- ✅ Validation on test flows (Taylor-Green, Lamb-Oseen)
- ✅ Pathological test cases (vortex filaments, potential singularities)
- ✅ Reproduction scripts for figures and results
- ✅ Complete mathematical documentation

##  Numerical Validation
Our approach combines:
- **High-precision spectral computation** (3D FFT)
- **Rigorous error estimation**
- **Comparison with analytical solutions**
- **Critical case testing** (Buckmaster-Vicol, etc.)

## 📜 License
MIT License - see LICENSE file for details.

##  Reference
This work accompanies the arXiv preprint: [link to be added]

## 👨‍💻 Author
Mushagalusa Bashizi Grâce
