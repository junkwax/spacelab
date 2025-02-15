# tests/test_spacetime.py
import pytest
import numpy as np
from src.models.spacetime import SpacetimeGeometry

def test_schwarzschild_metric():
    spacetime = SpacetimeGeometry(mass=10)
    g_tt, g_rr, *_ = spacetime.schwarzschild_metric(1e6)
    assert g_tt < 0
    assert g_rr > 0

def test_ricci_curvature():
    spacetime = SpacetimeGeometry(mass=10)
    assert np.all(spacetime.ricci_curvature([1e6, 1e7]) == 0)

def test_invalid_mass():
    with pytest.raises(ValueError):
        SpacetimeGeometry(mass=-10)