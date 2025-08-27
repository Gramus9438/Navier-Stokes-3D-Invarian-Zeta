# tests/test_physical_identities.py
import numpy as np
from src.zeta_calculator import NSConfig, NavierStokes3D, ic_beltrami_abc, run_simulation

DIV_TOL = 1e-6  # tolérance réaliste pour petits N + FFT

def test_incompressibility_preserved_after_projection():
    cfg = NSConfig(N=24, dt=6e-3, steps=0, nu=1e-2, save_every=1)
    sim = NavierStokes3D(cfg)
    sim.u_hat = ic_beltrami_abc(sim)  # champ solénoïdal (projeté)

    div = (sim.irfft3(1j*sim.kx*sim.u_hat[0]) +
           sim.irfft3(1j*sim.ky*sim.u_hat[1]) +
           sim.irfft3(1j*sim.kz*sim.u_hat[2]))

    m = float(np.abs(div).max())
    # aide au debug si ça rate en CI
    assert m < DIV_TOL, f"max|div u|={m:.3e} > {DIV_TOL:.0e}"

def test_energy_and_zeta_monitored_over_time():
    cfg = NSConfig(N=24, dt=6e-3, steps=15, nu=1e-2, save_every=1)
    out = run_simulation("taylor_green", ns_cfg=cfg)

    assert len(out["t"]) == 16
    assert np.isfinite(out["zeta"]).all()
    assert (out["zeta"] >= 0).all()
    assert (out["energy"] >= 0).all()

def test_short_step_stability():
    cfg = NSConfig(N=24, dt=5e-3, steps=1, nu=1e-2, save_every=1)
    out = run_simulation("beltrami", ns_cfg=cfg)
    for key in ["energy", "enstrophy", "H12", "zeta"]:
        assert np.isfinite(out[key]).all()
        assert (out[key] >= 0).all()
