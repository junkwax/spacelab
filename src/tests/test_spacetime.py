import pytest
import numpy as np
from src.models.spacetime import SpacetimeGeometry


def test_schwarzschild_metric():
    spacetime = SpacetimeGeometry(mass=10.0)
    r_test = 1e7
    g_tt, g_rr, g_theta_theta, g_phi_phi = spacetime.schwarzschild_metric(r_test)
    expected_g_tt = -0.997046

    assert np.isclose(g_tt, expected_g_tt, atol=1e-5)
    assert g_rr > 1.0
    assert np.isclose(g_theta_theta, r_test ** 2)


def test_schwarzschild_metric_nonequatorial():
    spacetime = SpacetimeGeometry(mass=10.0)
    r_test = 1e7

    _, _, _, g_phi_equator = spacetime.schwarzschild_metric(r_test, theta=np.pi / 2)
    _, _, _, g_phi_pole = spacetime.schwarzschild_metric(r_test, theta=0.0)

    assert np.isclose(g_phi_pole, 0.0, atol=1e-20)
    assert g_phi_equator > 0


def test_schwarzschild_rejects_inside_horizon():
    spacetime = SpacetimeGeometry(mass=10.0)
    rs = spacetime._get_schwarzschild_radius()
    with pytest.raises(ValueError, match="Schwarzschild radius"):
        spacetime.schwarzschild_metric(rs * 0.5)


def test_ricci_curvature_array():
    spacetime = SpacetimeGeometry(mass=10)
    r_vals = np.linspace(1e6, 1e7, 10)
    dilaton = np.zeros_like(r_vals)
    graviphoton = np.zeros_like(r_vals)

    R = spacetime.ricci_curvature(r_vals, dilaton, graviphoton)
    assert isinstance(R, np.ndarray)
    assert R.shape == r_vals.shape


def test_ricci_curvature_scalar():
    spacetime = SpacetimeGeometry(mass=10)
    R = spacetime.ricci_curvature(1e7, 0.0, 0.0)
    assert R != 0.0
    assert np.isfinite(R)
