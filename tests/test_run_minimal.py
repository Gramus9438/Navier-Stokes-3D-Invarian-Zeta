# tests/test_run_minimal.py
"""
Sanity test: run a very short simulation to check nothing crashes.
"""

from src import run_simulation, NSConfig

def test_taylor_green_short():
    cfg = NSConfig(N=16, steps=2, dt=0.01, nu=0.1, save_every=1)
    out = run_simulation("taylor_green", ns_cfg=cfg)
    # Vérifie qu'on a bien des séries avec au moins 3 points (steps=2 + initial)
    assert len(out["t"]) == 3
    # Vérifie que ζ(t) est bien fini et positif
    assert (out["zeta"] >= 0).all()
