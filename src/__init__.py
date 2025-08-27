# src/__init__.py
"""
Navier-Stokes-3D-Invariant-Zeta package.

Provides:
- NavierStokes3D: 3D pseudo-spectral solver
- NSConfig, ZetaConfig: configuration dataclasses
- run_simulation: helper to run a case
- IC_LIBRARY: library of initial conditions
"""

from .zeta_calculator import (
    NavierStokes3D,
    NSConfig,
    ZetaConfig,
    run_simulation,
    IC_LIBRARY,
)

__all__ = [
    "NavierStokes3D",
    "NSConfig",
    "ZetaConfig",
    "run_simulation",
    "IC_LIBRARY",
]
