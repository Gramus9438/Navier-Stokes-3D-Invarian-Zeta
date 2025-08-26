# tests/test_zeta.py
import numpy as np
import pytest
from src.zeta_calculator import NSConfig, run_simulation, IC_LIBRARY

@pytest.mark.parametrize("case", ["taylor_green", "beltrami"])
def test_runs_and_produces_finite_diagnostics(case):
    # config "rapide" pour tests CI
    cfg = NSConfig(N=24, dt=6e-3, steps=30, nu=1e-2, save_every=3)
    out = run_simulation(case, ns_cfg=cfg)

    # séries présentes et non vides
    for key in ["t", "energy", "enstrophy", "helicity", "H12", "zeta"]:
        assert key in out
        assert len(out[key]) > 0

    # toutes les valeurs sont finies
    for key in ["energy", "enstrophy", "helicity", "H12", "zeta"]:
        assert np.isfinite(out[key]).all()

    # quantités physiques non négatives
    assert (out["energy"] >= 0).all()
    assert (out["enstrophy"] >= 0).all()
    assert (out["H12"] >= 0).all()
    assert (out["zeta"] >= 0).all()

def test_ic_library_has_10_cases():
    # Vérifie qu'on expose bien les 10 cas
    assert len(IC_LIBRARY) == 10
