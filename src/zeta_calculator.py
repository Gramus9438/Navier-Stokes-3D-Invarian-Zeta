# src/zeta_calculator.py
"""
Navier-Stokes-3D-Invariant-Zeta
================================

3D pseudo-spectral Navier–Stokes solver (periodic box) with monitoring of ζ(t).

Features:
- Fourier pseudo-spectral method (periodic domain)
- Helmholtz projection for incompressibility
- 2/3 de-aliasing
- Diagnostics: energy, enstrophy, helicity, H^{1/2} norm
- ζ(t) invariant (default: sqrt( H^1/2^2 + beta * ||omega||_2^2 ))
- 10 canonical initial conditions (Beltrami, Taylor–Green, Lamb–Oseen, etc.)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

Array = np.ndarray


# =======================
# Fourier utilities
# =======================
def fourier_wavenumbers(N: int, L: float) -> Tuple[Array, Array, Array, Array]:
    """Return (kx, ky, kz, |k|^2) on a periodic box of size L with N^3 grid."""
    dx = L / N
    k1d = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    ks2 = kx**2 + ky**2 + kz**2
    return kx, ky, kz, ks2


def twothirds_dealias_mask(N: int) -> Array:
    """2/3-rule dealiasing mask in index space."""
    n1d = np.fft.fftfreq(N) * N
    nx, ny, nz = np.meshgrid(n1d, n1d, n1d, indexing="ij")
    cutoff = N // 3
    mask = (np.abs(nx) <= cutoff) & (np.abs(ny) <= cutoff) & (np.abs(nz) <= cutoff)
    return mask.astype(np.float64)


# =======================
# Configurations
# =======================
@dataclass
class NSConfig:
    N: int = 64
    L: float = 2 * np.pi
    nu: float = 1e-2
    dt: float = 5e-3
    steps: int = 200
    dealias: bool = True
    save_every: int = 10


@dataclass
class ZetaConfig:
    """
    Default:
      zeta = sqrt( ||u||_{H^{1/2}}^2 + beta * ||omega||_2^2 )
    You can override via custom_fn(diag_dict) -> float.
    """
    beta: float = 1.0
    custom_fn: Callable[[Dict[str, float]], float] | None = None


# =======================
# Solver
# =======================
class NavierStokes3D:
    def __init__(self, cfg: NSConfig, zcfg: ZetaConfig = ZetaConfig()):
        self.cfg = cfg
        self.zcfg = zcfg

        N, L = cfg.N, cfg.L
        self.N, self.L = N, L

        self.kx, self.ky, self.kz, self.ks2 = fourier_wavenumbers(N, L)
        self.ks2[0, 0, 0] = 1.0  # avoid division by zero at k=0
        self.mask = twothirds_dealias_mask(N) if cfg.dealias else 1.0

        # velocity Fourier components û = (ûx, ûy, ûz)
        self.u_hat = np.zeros((3, N, N, N), dtype=np.complex128)

        # diffusion implicit denominator (Backward Euler for viscosity)
        self.diff_denom = 1.0 + cfg.nu * cfg.dt * self.ks2

    # --- FFT helpers (only over spatial axes) ---
    def rfft3(self, a: Array) -> Array:
        # Transform only the last 3 axes (x,y,z); avoids mixing vector components.
        return np.fft.fftn(a, axes=(-3, -2, -1))

    def irfft3(self, a_hat: Array) -> Array:
        # Inverse transform only the last 3 axes (x,y,z)
        return np.fft.ifftn(a_hat, axes=(-3, -2, -1)).real

    # --- Incompressible projection (Helmholtz) in Fourier space ---
    def project(self, X_hat: Array) -> Array:
        k_dot_X = self.kx * X_hat[0] + self.ky * X_hat[1] + self.kz * X_hat[2]
        proj = np.empty_like(X_hat)
        proj[0] = X_hat[0] - self.kx * k_dot_X / self.ks2
        proj[1] = X_hat[1] - self.ky * k_dot_X / self.ks2
        proj[2] = X_hat[2] - self.kz * k_dot_X / self.ks2
        return proj

    # --- Nonlinear term (u·∇)u via pseudo-spectral ---
    def nonlinear_hat(self, u_hat: Array) -> Array:
        # velocity in physical space (transform per component)
        u = np.empty_like(u_hat, dtype=np.float64)
        u[0] = self.irfft3(u_hat[0])
        u[1] = self.irfft3(u_hat[1])
        u[2] = self.irfft3(u_hat[2])

        # gradients of each component in physical space
        dux_dx = self.irfft3(1j * self.kx * u_hat[0])
        dux_dy = self.irfft3(1j * self.ky * u_hat[0])
        dux_dz = self.irfft3(1j * self.kz * u_hat[0])

        duy_dx = self.irfft3(1j * self.kx * u_hat[1])
        duy_dy = self.irfft3(1j * self.ky * u_hat[1])
        duy_dz = self.irfft3(1j * self.kz * u_hat[1])

        duz_dx = self.irfft3(1j * self.kx * u_hat[2])
        duz_dy = self.irfft3(1j * self.ky * u_hat[2])
        duz_dz = self.irfft3(1j * self.kz * u_hat[2])

        # advective term components
        adv_x = u[0] * dux_dx + u[1] * dux_dy + u[2] * dux_dz
        adv_y = u[0] * duy_dx + u[1] * duy_dy + u[2] * duy_dz
        adv_z = u[0] * duz_dx + u[1] * duz_dy + u[2] * duz_dz

        adv_hat = np.empty_like(u_hat)
        adv_hat[0] = self.rfft3(adv_x)
        adv_hat[1] = self.rfft3(adv_y)
        adv_hat[2] = self.rfft3(adv_z)

        # dealias
        adv_hat *= self.mask

        # project to divergence-free
        adv_hat = self.project(adv_hat)

        # sign: -(u·∇)u
        return -adv_hat

    # --- One time step: explicit NL + implicit diffusion ---
    def step(self):
        Nhat = self.nonlinear_hat(self.u_hat)
        self.u_hat = (self.u_hat + self.cfg.dt * Nhat) / self.diff_denom
        # light stabilization (dealias again)
        if isinstance(self.mask, np.ndarray):
            self.u_hat[0] *= self.mask
            self.u_hat[1] *= self.mask
            self.u_hat[2] *= self.mask

    # --- Diagnostics ---
    def velocity(self) -> Array:
        u = np.empty_like(self.u_hat, dtype=np.float64)
        u[0] = self.irfft3(self.u_hat[0])
        u[1] = self.irfft3(self.u_hat[1])
        u[2] = self.irfft3(self.u_hat[2])
        return u

    def vorticity_hat(self) -> Array:
        wx = 1j * (self.ky * self.u_hat[2] - self.kz * self.u_hat[1])
        wy = 1j * (self.kz * self.u_hat[0] - self.kx * self.u_hat[2])
        wz = 1j * (self.kx * self.u_hat[1] - self.ky * self.u_hat[0])
        return np.array([wx, wy, wz])

    def vorticity(self) -> Array:
        w_hat = self.vorticity_hat()
        w = np.empty_like(w_hat, dtype=np.float64)
        w[0] = self.irfft3(w_hat[0])
        w[1] = self.irfft3(w_hat[1])
        w[2] = self.irfft3(w_hat[2])
        return w

    def energy(self) -> float:
        u = self.velocity()
        return 0.5 * float(np.mean(u[0] ** 2 + u[1] ** 2 + u[2] ** 2))

    def enstrophy(self) -> float:
        w = self.vorticity()
        return 0.5 * float(np.mean(w[0] ** 2 + w[1] ** 2 + w[2] ** 2))

    def helicity(self) -> float:
        u = self.velocity()
        w = self.vorticity()
        return float(np.mean(u[0] * w[0] + u[1] * w[1] + u[2] * w[2]))

    def Hs_norm(self, s: float = 0.5) -> float:
        # ||u||_{H^s} ~ sqrt( mean( (1+|k|^2)^s |û|^2 ) )
        weight = (1.0 + self.ks2) ** s
        val = (weight * (np.abs(self.u_hat[0]) ** 2 +
                         np.abs(self.u_hat[1]) ** 2 +
                         np.abs(self.u_hat[2]) ** 2)).mean()
        return float(np.sqrt(val))

    def zeta(self) -> float:
        diag = {
            "energy": self.energy(),
            "enstrophy": self.enstrophy(),
            "helicity": self.helicity(),
            "H12": self.Hs_norm(0.5),
        }
        if self.zcfg.custom_fn is not None:
            return float(self.zcfg.custom_fn(diag))
        # default definition: sqrt( H^{1/2}^2 + beta * ||omega||_2^2 )
        return float(np.sqrt(diag["H12"] ** 2 + self.zcfg.beta * 2.0 * diag["enstrophy"]))


# =======================
# Initial conditions (10)
# =======================
def _fft_vec(u: Array, fftfn: Callable[[Array], Array]) -> Array:
    """Real vector field (3,N,N,N) -> Fourier (3,N,N,N)."""
    return np.array([fftfn(u[0]), fftfn(u[1]), fftfn(u[2])])

# 1) Beltrami (ABC flow)
def ic_beltrami_abc(sim: NavierStokes3D, A=1.0, B=1.0, C=1.0, k0=1) -> Array:
    N, L = sim.N, sim.L
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    u = np.zeros((3, N, N, N), dtype=np.float64)
    u[0] = A*np.sin(k0*Z) + C*np.cos(k0*Y)
    u[1] = B*np.sin(k0*X) + A*np.cos(k0*Z)
    u[2] = C*np.sin(k0*Y) + B*np.cos(k0*X)
    return _fft_vec(u, sim.rfft3)

# 2) Taylor–Green vortex
def ic_taylor_green(sim: NavierStokes3D, U0=1.0, k0=1) -> Array:
    N, L = sim.N, sim.L
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.zeros((3, N, N, N))
    u[0] =  U0*np.sin(k0*X)*np.cos(k0*Y)*np.cos(k0*Z)
    u[1] = -U0*np.cos(k0*X)*np.sin(k0*Y)*np.cos(k0*Z)
    u[2] = 0.0
    return _fft_vec(u, sim.rfft3)

# helper: 2D Lamb–Oseen velocity at time t0
def _lamb_oseen_velocity(x, y, nu, t0, Gamma=1.0):
    r2 = x**2 + y**2 + 1e-30
    core = 4.0 * nu * t0
    u_theta = Gamma/(2*np.pi*np.sqrt(r2)) * (1.0 - np.exp(-r2/core))
    ux = -u_theta * (y/np.sqrt(r2))
    uy =  u_theta * (x/np.sqrt(r2))
    return ux, uy

# 3) Lamb–Oseen (extended 2D field in 3D)
def ic_lamb_oseen(sim: NavierStokes3D, Gamma=1.0, t0=0.05) -> Array:
    N, L = sim.N, sim.L
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    ux, uy = _lamb_oseen_velocity(X, Y, sim.cfg.nu, t0, Gamma=Gamma)
    u = np.zeros((3, N, N, N))
    u[0], u[1], u[2] = ux, uy, 0.0
    return _fft_vec(u, sim.rfft3)

# 4) Compact bump (localized smooth field)
def ic_compact_bump(sim: NavierStokes3D) -> Array:
    N, L = sim.N, sim.L
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    r2 = X**2 + Y**2 + Z**2
    psi = np.exp(- r2 / (0.3*L)**2 )
    dpsi_dx = np.gradient(psi, x, axis=0)
    dpsi_dy = np.gradient(psi, x, axis=1)
    u = np.zeros((3,N,N,N))
    u[0] = dpsi_dy
    u[1] = -dpsi_dx
    u[2] = 0.0
    return _fft_vec(u, sim.rfft3)

# 5) Rough H^{1/2} (random solenoidal with decaying spectrum)
def ic_rough_H12(sim: NavierStokes3D, seed=0) -> Array:
    rng = np.random.default_rng(seed)
    shape = (3, sim.N, sim.N, sim.N)
    rnd = (rng.standard_normal(shape) + 1j*rng.standard_normal(shape))
    amp = (1.0 + sim.ks2)**(-0.75)  # ~ |k|^{-1.5} -> H^{1/2}
    u_hat = rnd * amp
    u_hat = sim.project(u_hat)
    return u_hat

# 6) Axisymmetric swirl
def ic_axisymmetric_swirl(sim: NavierStokes3D, Omega=1.0) -> Array:
    N, L = sim.N, sim.L
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    R2 = X**2 + Y**2
    gaussian = np.exp(-R2/(0.25*L)**2)
    u = np.zeros((3,N,N,N))
    u[0] = -Omega*Y * gaussian
    u[1] =  Omega*X * gaussian
    u[2] =  0.0
    return _fft_vec(u, sim.rfft3)

# 7) Small scales superposition (random truncated Fourier content)
def ic_small_scale_superposition(sim: NavierStokes3D, kmin=8, kmax=12, seed=1) -> Array:
    rng = np.random.default_rng(seed)
    u_hat = np.zeros_like(sim.u_hat, dtype=np.complex128)
    for _ in range(200):
        kx = rng.integers(-kmax, kmax+1)
        ky = rng.integers(-kmax, kmax+1)
        kz = rng.integers(-kmax, kmax+1)
        if max(abs(kx),abs(ky),abs(kz)) < kmin or (kx==ky==kz==0):
            continue
        phase = np.exp(1j * 2*np.pi * rng.random())
        ix = (kx % sim.N)
        iy = (ky % sim.N)
        iz = (kz % sim.N)
        vec = rng.standard_normal(3) + 1j*rng.standard_normal(3)
        u_hat[:, ix, iy, iz] += 1e-2 * phase * vec
    return sim.project(u_hat)

# 8) Truncated jet (mean flow + small-scale perturbations)
def ic_truncated_jet(sim: NavierStokes3D, U=1.5) -> Array:
    N, L = sim.N, sim.L
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    r = np.sqrt(X**2 + Y**2)
    profile = U * np.exp(- (r/(0.15*L))**2 )
    u = np.zeros((3,N,N,N))
    u[2] = profile
    pert = ic_small_scale_superposition(sim, kmin=6, kmax=10)
    u_hat = _fft_vec(u, sim.rfft3) + 0.15*pert
    return sim.project(u_hat)

# 9) Shear layer
def ic_shear_layer(sim: NavierStokes3D, U=1.0) -> Array:
    N, L = sim.N, sim.L
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    delta = 0.08*L
    u = np.zeros((3,N,N,N))
    u[0] = U*np.tanh(Y/delta)
    u[1] = 0.05*U*np.sin(2*np.pi*Z/L)*np.exp(-(Y/delta)**2)
    u[2] = 0.0
    return sim.project(_fft_vec(u, sim.rfft3))

# 10) Double opposite vortices (two Lamb–Oseen with opposite circulation)
def ic_double_opposite_vortex(sim: NavierStokes3D, sep=0.4, Gamma=1.0, t0=0.05) -> Array:
    N, L = sim.N, sim.L
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    x1, x2 = -sep*L/2, sep*L/2
    ux1, uy1 = _lamb_oseen_velocity(X-x1, Y, sim.cfg.nu, t0, Gamma=+Gamma)
    ux2, uy2 = _lamb_oseen_velocity(X-x2, Y, sim.cfg.nu, t0, Gamma=-Gamma)
    u = np.zeros((3,N,N,N))
    u[0] = ux1 + ux2
    u[1] = uy1 + uy2
    u[2] = 0.0
    return _fft_vec(u, sim.rfft3)


# Map: case name -> generator
IC_LIBRARY: Dict[str, Callable[[NavierStokes3D], Array]] = {
    "beltrami": ic_beltrami_abc,                    # 1
    "taylor_green": ic_taylor_green,                # 2
    "lamb_oseen": ic_lamb_oseen,                    # 3
    "compact_bump": ic_compact_bump,                # 4
    "rough_H12": ic_rough_H12,                      # 5
    "axisymmetric_swirl": ic_axisymmetric_swirl,    # 6
    "truncated_jet": ic_truncated_jet,              # 7
    "shear_layer": ic_shear_layer,                  # 8
    "small_scales": ic_small_scale_superposition,   # 9
    "double_opposite": ic_double_opposite_vortex    # 10
}


# =======================
# Runner
# =======================
def run_simulation(
    case: str,
    ns_cfg: NSConfig = NSConfig(),
    z_cfg: ZetaConfig = ZetaConfig(),
    ic_kwargs: Dict | None = None,
    log_callback: Callable[[int, Dict[str, float]], None] | None = None,
) -> Dict[str, Array]:
    """Run a simulation and return diagnostics time series."""
    ic_kwargs = ic_kwargs or {}
    sim = NavierStokes3D(ns_cfg, z_cfg)
    if case not in IC_LIBRARY:
        raise ValueError(f"Unknown case: {case}. Options: {list(IC_LIBRARY.keys())}")
    sim.u_hat = IC_LIBRARY[case](sim, **ic_kwargs)

    times, energy, enst, helic, H12, zeta = [], [], [], [], [], []
    t = 0.0
    for n in range(ns_cfg.steps + 1):
        if n % ns_cfg.save_every == 0:
            e = sim.energy()
            en = sim.enstrophy()
            he = sim.helicity()
            h12 = sim.Hs_norm(0.5)
            zz = sim.zeta()
            times.append(t); energy.append(e); enst.append(en); helic.append(he); H12.append(h12); zeta.append(zz)
            if log_callback:
                log_callback(n, {"t": t, "energy": e, "enstrophy": en, "helicity": he, "H12": h12, "zeta": zz})
        sim.step()
        t += ns_cfg.dt

    return {
        "t": np.array(times),
        "energy": np.array(energy),
        "enstrophy": np.array(enst),
        "helicity": np.array(helic),
        "H12": np.array(H12),
        "zeta": np.array(zeta),
    }


# =======================
# Direct execution (quick smoke test)
# =======================
if __name__ == "__main__":
    cfg = NSConfig(N=48, dt=4e-3, steps=200, nu=1e-2, save_every=5)
    out = run_simulation("taylor_green", cfg)
    print("Final diagnostics:", {k: float(v[-1]) for k, v in out.items() if k != "t"})
