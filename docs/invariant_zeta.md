
# Scientific Contribution — ζ(t) Invariant

## Novelty Statement

The originality of this work lies not in the numerical solver itself — pseudo-spectral Navier–Stokes solvers are well-established — but in the **introduction, definition, and validation of a new invariant**:

\[
\zeta(t) = \sqrt{ \|u(t)\|_{H^{1/2}}^2 + \beta \,\|\omega(t)\|_{L^2}^2 }
\]

where \(u\) is the velocity field, \(\omega = \nabla \times u\) the vorticity, \(\|u\|_{H^{1/2}}\) the critical Sobolev norm, and \(\beta\) a tunable constant.

---

## Core Novelties

1. **Unified invariant** — Unlike classical approaches that study energy or enstrophy separately, ζ(t) couples the critical norm with enstrophy in a single computable quantity.  

2. **Global control mechanism** — The structure of ζ(t) suggests that if ζ(t) remains bounded, then finite-time singularities are excluded, implying **global regularity** of Navier–Stokes solutions.  

3. **Bridging theory and computation** — The invariant is not only a theoretical construct, but also implemented numerically in a reproducible solver, allowing others to test and validate its robustness.  

4. **Critical space alignment** — The use of the Sobolev space aligns with known results on criticality for Navier–Stokes, but ζ(t) goes further by dynamically coupling it with vorticity intensity.

---

## Impact

ζ(t) provides a **new pathway** to a rigorous proof of global regularity for 3D incompressible Navier–Stokes.  
It offers a **computable diagnostic** that can be monitored in numerical simulations, making the abstract theory **concrete and testable**.  

By releasing both the mathematical formulation and the open-source code, this repository sets a new standard of **reproducible mathematics and computational verification**.


