import pytest
import numpy as np
from src.models.spacetime import SpacetimeGeometry

def test_schwarzschild_metric():
    # Test that the Schwarzschild metric is calculated correctly.
    spacetime = SpacetimeGeometry(mass=10)
    g_tt, g_rr, g_theta_theta, g_phi_phi = spacetime.schwarzschild_metric(1e7)
    assert np.isclose(g_tt, -0.9999985420000001)
    assert np.isclose(g_rr, 1.000001458)
    assert np.isclose(g_theta_theta, 1e14)
    assert np.isclose(g_phi_phi, 1e14)

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