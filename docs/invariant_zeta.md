# Invariant ζ(t) — Scientific Contribution

## 1. Definition

We introduce the invariant:

\[
\zeta(t) = \sqrt{ \|u(t)\|_{H^{1/2}}^2 + \beta \,\|\omega(t)\|_{L^2}^2 }
\]

where:  
- \(u(t)\) is the velocity field of the 3D incompressible Navier–Stokes equations,  
- \(\omega = \nabla \times u\) is the vorticity,  
- \(\|u\|_{H^{1/2}}\) is the critical Sobolev norm,  
- \(\beta > 0\) is a tunable parameter.

---

## 2. Novelty Statement

The originality of this work lies not in the numerical solver itself — pseudo-spectral Navier–Stokes solvers are well-established — but in the **introduction, definition, and validation of a new invariant ζ(t)**.

---

## 3. Core Novelties

1. **Unified invariant**  
   Unlike classical approaches that monitor energy or enstrophy separately, ζ(t) couples the **critical Sobolev norm** with **enstrophy** in a single, computable, dynamic quantity.  

2. **Global control mechanism**  
   The structure of ζ(t) suggests that if ζ(t) remains bounded, then finite-time singularities are excluded. This provides a theoretical pathway to **global regularity** of Navier–Stokes solutions.  

3. **Bridge between theory and computation**  
   ζ(t) is not only a theoretical construct, but also **implemented in reproducible code**. This allows others to test, validate, and stress the invariant against diverse flows and potential singular scenarios.  

4. **Critical space alignment**  
   The use of the Sobolev \(H^{1/2}\) space is consistent with the criticality of Navier–Stokes. ζ(t) extends this by dynamically coupling the critical Sobolev structure with **vorticity intensity**, creating a more robust diagnostic.  

---

## 4. Impact

- ζ(t) provides a **new invariant-based framework** to approach one of the Clay Millennium Problems: global regularity for 3D incompressible Navier–Stokes.  
- It offers a **computable diagnostic** that can be monitored in simulations, bridging the gap between **abstract PDE theory** and **practical numerics**.  
- By releasing both the **mathematical formulation** and the **open-source implementation**, this repository establishes a **reproducible standard** for mathematical verification through computation.  

---

## 5. References

- Clay Mathematics Institute — Millennium Problems.  
- Classical spectral methods for Navier–Stokes (Orszag, Canuto, et al.).  
- Critical space approaches to NS regularity (Koch & Tataru, Tao, etc.).
