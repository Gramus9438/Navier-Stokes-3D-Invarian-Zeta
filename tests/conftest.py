# tests/conftest.py
"""
Pytest configuration file to ensure the repository root is on sys.path.

This guarantees that imports like:
    from src.zeta_calculator import NavierStokes3D
work correctly both locally and in CI.
"""

import sys
import pathlib

# Resolve the path of the repo root (one level up from /tests)
ROOT = pathlib.Path(__file__).resolve().parents[1]

# Insert it at the beginning of sys.path if not already there
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
