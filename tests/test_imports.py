# tests/test_imports.py
"""
Smoke test to ensure the src package is importable
and exposes the expected API.
"""

import pytest

def test_import_src():
    import src
    # Check that key symbols exist
    assert hasattr(src, "NavierStokes3D")
    assert hasattr(src, "NSConfig")
    assert hasattr(src, "ZetaConfig")
    assert hasattr(src, "run_simulation")
    assert hasattr(src, "IC_LIBRARY")

def test_run_simulation_taylor_green():
    from src import NSConfig, run_simulation
    cfg = NSConfig(N=8, steps=1, dt=0.01, nu=0.1, save_every=1)
    out = run_simulation("taylor_green", cfg)
    # Basic sanity checks
    assert "zeta" in out
    assert len(out["t"]) >= 1
    assert (out["zeta"] >= 0).all()
