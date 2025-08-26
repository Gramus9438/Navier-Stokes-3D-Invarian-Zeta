
# Invariant ζ(t)

##  Background

The three–dimensional incompressible Navier–Stokes equations are one of the **Millennium Prize Problems** posed by the Clay Mathematics Institute.  
The central question is whether, for all smooth initial data with finite energy, the solutions remain **regular for all time**, or whether **finite–time singularities** may occur.

---

##  Innovation: the ζ(t) Invariant

We introduce a new invariant, denoted **ζ(t)**, designed to provide global control of solution growth:

$$
\zeta(t) = \sqrt{ \, \|u(t)\|_{H^{1/2}}^2 + \beta \, \|\omega(t)\|_{L^2}^2 \, }
$$

where:

- \(u(t)\) is the velocity field,  
- \(\omega = \nabla \times u\) is the vorticity field,  
- \(\|u\|_{H^{1/2}}\) is the critical Sobolev norm,  
- \(\beta > 0\) is a tunable parameter.

---

##  Theoretical Motivation

1. **Coupling critical scales**: ζ(t) links the critical Sobolev regularity \(H^{1/2}\) with vorticity intensity.  
2. **Singularity prevention**: the boundedness of ζ(t) prevents finite–time blow–up of energy or enstrophy.  
3. **Theory–numerics bridge**: unlike abstract invariants, ζ(t) is **computable in simulations**, making it verifiable.  
4. **Consistency with known results**: the Sobolev space \(H^{1/2}\) is known as critical for Navier–Stokes; ζ(t) introduces a new dynamic coupling with vorticity.

---

## 📊 Numerical Validation

The ζ(t) invariant is monitored in all our 3D pseudo–spectral simulations.  
The benchmark test cases include:

- **Classical flows**: Beltrami (ABC), Taylor–Green, Lamb–Oseen.  
- **Pathological flows**: vortex filaments, rough \(H^{1/2}\)–type initial data.  
- **Jets and shear layers**: unstable profiles generating strong gradients.  
- **Multi–scale superpositions**: small–scale excitations and nonlinear interactions.  

In all these cases, ζ(t) remains bounded and exhibits regular evolution.

---

##  Expected Impact

- If ζ(t) is **mathematically proven** to remain bounded, it provides a **proof of global regularity** for the 3D Navier–Stokes equations.  
- It becomes a **universal diagnostic** that can be verified by any researcher.  
- It establishes a framework for **reproducible mathematics**: the theory is accompanied by open–source, numerically validated code.  

---

## 📚 Reference

This work contributes to the attempted resolution of the Millennium Prize Problem:  
**Global Regularity for 3D Navier–Stokes Equations (Clay Institute).**

---

 Author: **Mushagalusa Bashizi Grâce**
