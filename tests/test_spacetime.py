import pytest
import numpy as np
from src.models.spacetime import SpacetimeGeometry

def test_schwarzschild_metric():
    spacetime = SpacetimeGeometry(mass=10)  # mass in solar masses
    
    # Choose a radius much larger than the Schwarzschild radius
    r = 1e6  # radius in meters, significantly larger than Schwarzschild radius for 10 solar masses
    
    g_tt, g_rr, *_ = spacetime.schwarzschild_metric(r)
    
    assert g_tt < 0  # g_tt should be negative
    assert g_rr > 0  # g_rr should be positive

def test_ricci_curvature():
    spacetime = SpacetimeGeometry(mass=10)
    assert np.all(spacetime.ricci_curvature([1e6, 1e7]) == 0)

def test_invalid_mass():
    with pytest.raises(ValueError):
        SpacetimeGeometry(mass=-10)
