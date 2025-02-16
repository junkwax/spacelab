import pytest
import numpy as np
from src.models.spacetime import SpacetimeGeometry

def test_schwarzschild_metric():
    spacetime = SpacetimeGeometry(mass=10)
    r = 1e7
    g_tt, g_rr, g_theta_theta, g_phi_phi = spacetime.schwarzschild_metric(r)
    
    # Compute the correct expected values
    rs = 2 * spacetime.G * (spacetime.mass * 1.98847e30) / (spacetime.c**2)
    expected_g_tt = -(1 - rs / r)
    expected_g_rr = 1 / (1 - rs / r)

    assert np.isclose(g_tt, expected_g_tt, atol=1e-6)  # Use computed expected value
    assert np.isclose(g_rr, expected_g_rr, atol=1e-6)
    assert np.isclose(g_theta_theta, r**2)
    assert np.isclose(g_phi_phi, r**2)

def test_ricci_curvature():
    # Test that the Ricci curvature is calculated correctly.
    spacetime = SpacetimeGeometry(mass=10)
    # Provide values for dilaton_field and graviphoton_field
    dilaton_field = 0.1  # Example value
    graviphoton_field = 0.0  # Example value
    # Call ricci_curvature with all required arguments
    R = spacetime.ricci_curvature([1e6, 1e7], dilaton_field, graviphoton_field)
    # Assert that the result is close to zero (or the expected value based on your model)
    assert np.allclose(R, 0, atol=1e-6)  # Adjust tolerance as needed