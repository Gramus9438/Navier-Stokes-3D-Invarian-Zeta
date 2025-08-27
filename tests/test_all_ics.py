# tests/test_all_ics.py
"""
Test all initial conditions (IC_LIBRARY):
- Each IC generates a valid divergence-free field in Fourier space
- A 1-step simulation runs without errors and produces finite diagnostics
"""

import numpy as np
import pytest

from src import IC_LIBRARY, NSConfig, ZetaConfig, NavierStokes3D, run_simulation


@pytest.mark.parametrize("case", list(IC_LIBRARY.keys()))
def test_ic_generation_and_divergence(case):
    # Small grid to keep CI fast
    cfg = NSConfig(N=12, dt=1e-2, steps=0, nu=1e-2, save_every=1, dealias=True)
    sim = NavierStokes3D(cfg, ZetaConfig())

    # Generate initial condition û(k)
    u_hat = IC_LIBRARY[case](sim)
    assert isinstance(u_hat, np.ndarray)
    assert u_hat.shape == (3, cfg.N, cfg.N, cfg.N)

    # Finiteness
    assert np.isfinite(u_hat.real).all()
    assert np.isfinite(u_hat.imag).all()

    # Check incompressibility in real space: div u ≈ 0
    div = (
        sim.irfft3(1j * sim.kx * u_hat[0])
        + sim.irfft3(1j * sim.ky * u_hat[1])
        + sim.irfft3(1j * sim.kz * u_hat[2])
    )
    # Tolerance generous to avoid FP noise on small grids
    assert float(np.abs(div).max()) < 1e-8


@pytest.mark.parametrize("case", list(IC_LIBRARY.keys()))
def test_one_step_run(case):
    # 1 step run to ensure pipeline is consistent for all ICs
    cfg = NSConfig(N=12, dt=5e-3, steps=1, nu=1e-2, save_every=1, dealias=True)
    out = run_simulation(case, ns_cfg=cfg)

    # Time series lengths: steps=1 and save_every=1 -> 2 points (t=0 and t=dt)
    assert len(out["t"]) == 2

    # Finite, non-negative key diagnostics
    for key in ["energy", "enstrophy", "H12", "zeta"]:
        assert np.isfinite(out[key]).all()
        assert (out[key] >= 0).all()

    # Helicity can be signed; only check finiteness
    assert np.isfinite(out["helicity"]).all()
