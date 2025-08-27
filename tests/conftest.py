# tests/conftest.py
import sys, pathlib
# add repo root to sys.path so "from src..." works everywhere
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
