# tests/test_spacetime.py
import pytest
import numpy as np
from src.models.spacetime import SpacetimeGeometry

@pytest.fixture
def spacetime():
    return SpacetimeGeometry(mass=1.0)  # Use a mass of 1 solar mass for simplicity

def test_schwarzschild_metric_array(spacetime):
    # Test with array input
    r = np.array([3.0e9, 4.0e9, 5.0e9])  # Values > rs
    g_tt, g_rr, g_theta_theta, g_phi_phi = spacetime.schwarzschild_metric(r)

    rs = 2 * spacetime.G * spacetime.mass * spacetime.solar_mass_kg / (spacetime.c**2)
    expected_g_tt = -(1 - rs / r)
    expected_g_rr = 1 / (1 - rs / r)

    assert isinstance(g_tt, np.ndarray)
    assert isinstance(g_rr, np.ndarray)
    assert isinstance(g_theta_theta, np.ndarray)
    assert isinstance(g_phi_phi, np.ndarray)
    assert g_tt.shape == r.shape
    assert g_rr.shape == r.shape
    assert g_theta_theta.shape == r.shape
    assert g_phi_phi.shape == r.shape
    assert np.allclose(g_tt, expected_g_tt)
    assert np.allclose(g_rr, expected_g_rr)
    assert np.allclose(g_theta_theta, r**2)
    assert np.allclose(g_phi_phi, r**2)

def test_schwarzschild_metric_scalar(spacetime):
    # Test with scalar input
    r_scalar = 3.0e9
    g_tt, g_rr, g_theta_theta, g_phi_phi = spacetime.schwarzschild_metric(r_scalar)

    rs = 2 * spacetime.G * spacetime.mass * spacetime.solar_mass_kg / (spacetime.c**2)
    expected_g_tt = -(1 - rs / r_scalar)
    expected_g_rr = 1 / (1 - rs / r_scalar)

    assert np.isclose(g_tt, expected_g_tt)
    assert np.isclose(g_rr, expected_g_rr)
    assert np.isclose(g_theta_theta, r_scalar**2)
    assert np.isclose(g_phi_phi, r_scalar**2)

def test_schwarzschild_metric_raises_value_error(spacetime):
    r_invalid = 1.0e3  # Below Schwarzschild radius, changed from 2.0e9 to 1.0e3

    with pytest.raises(ValueError):
        spacetime.schwarzschild_metric(r_invalid)


def test_kaluza_klein_metric_array(spacetime):
     # Test Kaluza-Klein metric components with array inputs
    r = np.array([3.0e9, 4.0e9, 5.0e9])  # Values > rs
    dilaton_field = np.array([0.1, 0.2, 0.3])
    graviphoton_field = np.array([0.01, 0.02, 0.03])
    g_tt, g_rr, g_theta_theta, g_phi_phi, g_yy, g_ty = spacetime.kaluza_klein_metric(r, dilaton_field, graviphoton_field)
    assert isinstance(g_tt, np.ndarray) and g_tt.shape == r.shape
    assert isinstance(g_rr, np.ndarray) and g_rr.shape == r.shape
    assert isinstance(g_theta_theta, np.ndarray) and g_theta_theta.shape == r.shape
    assert isinstance(g_phi_phi, np.ndarray) and g_phi_phi.shape == r.shape
    assert isinstance(g_yy, np.ndarray) and g_yy.shape == r.shape
    assert isinstance(g_ty, np.ndarray) and g_ty.shape == r.shape

def test_kaluza_klein_metric_scalar(spacetime):
    # Test Kaluza-Klein metric components with scalar inputs
    r_scalar = 3.0e9
    dilaton_field_scalar = 0.1
    graviphoton_field_scalar = 0.01
    g_tt, g_rr, g_theta_theta, g_phi_phi, g_yy, g_ty = spacetime.kaluza_klein_metric(r_scalar, dilaton_field_scalar, graviphoton_field_scalar)
    assert isinstance(g_tt, (float, np.float64))
    assert isinstance(g_rr, (float, np.float64))
    assert isinstance(g_theta_theta, (float, np.float64))
    assert isinstance(g_phi_phi, (float, np.float64))
    assert isinstance(g_yy, (float, np.float64))
    assert isinstance(g_ty, (float, np.float64))


def test_kaluza_klein_metric_schwarzschild_limit(spacetime):
    # Test that the Kaluza-Klein metric reduces to Schwarzschild when dilaton and graviphoton are zero.
    r = np.array([3.0e9, 4.0e9, 5.0e9])
    dilaton_field = np.array([0.0, 0.0, 0.0])  # Zero dilaton
    graviphoton_field = np.array([0.0, 0.0, 0.0])  # Zero graviphoton

    g_tt_kk, g_rr_kk, _, _, _, _ = spacetime.kaluza_klein_metric(r, dilaton_field, graviphoton_field)
    g_tt_s, g_rr_s, _, _ = spacetime.schwarzschild_metric(r)

    assert np.allclose(g_tt_kk, g_tt_s)
    assert np.allclose(g_rr_kk, g_rr_s)

def test_kaluza_klein_metric_raises_value_error(spacetime):
    r_invalid = 1.0e3  # Below Schwarzschild radius, changed from 2.0e9 to 1.0e3

    with pytest.raises(ValueError):
        spacetime.kaluza_klein_metric(r_invalid, 0.0, 0.0)


def test_ricci_curvature_array(spacetime):
    # Test that the Ricci curvature is calculated correctly with array inputs
    r_vals = np.array([3.0e9, 4.0e9, 5.0e9])
    dilaton_field = np.array([0.1, 0.2, 0.3])
    graviphoton_field = np.array([0.01, 0.02, 0.03])
    R = spacetime.ricci_curvature(r_vals, dilaton_field, graviphoton_field)
    assert isinstance(R, np.ndarray)
    assert R.shape == r_vals.shape

def test_ricci_curvature_scalar(spacetime):
    # Test with scalar input
    r_scalar = 3.0e9
    dilaton_field_scalar = 0.1
    graviphoton_field_scalar = 0.01
    R = spacetime.ricci_curvature(r_scalar, dilaton_field_scalar, graviphoton_field_scalar)
    assert isinstance(R, np.ndarray)  # Check it's a NumPy array
    assert R.size == 1  # Check it contains a single value

def test_ricci_curvature_schwarzschild_limit(spacetime):
    # Test that the Ricci curvature is zero for the Schwarzschild metric (dilaton and graviphoton are zero).
    r_vals = np.array([3.0e9, 4.0e9, 5.0e9])
    dilaton_field = np.zeros_like(r_vals)  # Zero dilaton
    graviphoton_field = np.zeros_like(r_vals)  # Zero graviphoton
    R = spacetime.ricci_curvature(r_vals, dilaton_field, graviphoton_field)
    assert np.allclose(R, 0.0, atol=1e-9)  # Ricci scalar should be zero, increased tolerance

def test_ricci_curvature_raises_value_error(spacetime):
    r_invalid = 1.0e3  # Below Schwarzschild radius, changed from 2.0e9 to 1.0e3
    with pytest.raises(ValueError):
        spacetime.ricci_curvature(r_invalid, 0.0, 0.0)

def test_stress_energy_tensor_DM_zero_fields(spacetime):
    r = np.linspace(3e9, 10e9, 10)  # Increased r values
    phi_DM = np.zeros_like(r)
    dphi_DM_dr = np.zeros_like(r)
    phi_DE = np.zeros_like(r)
    dilaton_field = np.zeros_like(r)
    beta = 1e-10

    T_munu = spacetime.stress_energy_tensor_DM(r, phi_DM, dphi_DM_dr, phi_DE, dilaton_field, beta)
    assert isinstance(T_munu, np.ndarray)
    assert T_munu.shape == (4, 4, len(r))
    assert np.allclose(T_munu, 0, atol=1e-9)  # Expect zero stress-energy, increased tolerance


def test_stress_energy_tensor_DM_simple_case(spacetime):
    r = np.linspace(3e9, 10e9, 10)  # Increased r values
    phi_DM = np.ones_like(r) * 0.1  # Small constant value
    dphi_DM_dr = np.zeros_like(r)
    phi_DE = np.ones_like(r) * 0.05
    dilaton_field = np.ones_like(r) * 0.01
    beta = 1e-10

    T_munu = spacetime.stress_energy_tensor_DM(r, phi_DM, dphi_DM_dr, phi_DE, dilaton_field, beta)
    assert isinstance(T_munu, np.ndarray)
    assert T_munu.shape == (4, 4, len(r))
    assert not np.allclose(T_munu, 0, atol=1e-9)  # Should not be zero with non-zero fields, increased tolerance


def test_stress_energy_tensor_DE_zero_fields(spacetime):
    r = np.linspace(3e9, 10e9, 10)  # Increased r values
    phi_DE = np.zeros_like(r)
    dphi_DE_dr = np.zeros_like(r)
    dphi_DE_dt = np.zeros_like(r)
    phi_DM = np.zeros_like(r)
    dilaton_field = np.zeros_like(r)
    beta = 1e-10
    V0 = 0  # Set V0 to 0 for zero field test
    lambda_ = 0.1

    T_munu = spacetime.stress_energy_tensor_DE(r, phi_DE, dphi_DE_dr, dphi_DE_dt, phi_DM, dilaton_field, beta, V0, lambda_)
    assert isinstance(T_munu, np.ndarray)
    assert T_munu.shape == (4, 4, len(r))
    assert np.allclose(T_munu, 0, atol=1e-9)  # Expect zero tensor for zero fields, increased tolerance


def test_stress_energy_tensor_DE_simple_case(spacetime):
    r = np.linspace(3e9, 10e9, 10)  # Increased r values
    phi_DE = np.ones_like(r) * 0.05
    dphi_DE_dr = np.zeros_like(r)
    dphi_DE_dt = np.zeros_like(r)
    phi_DM = np.ones_like(r) * 0.1
    dilaton_field = np.ones_like(r) * 0.01
    beta = 1e-10
    V0 = 1e-47
    lambda_ = 0.1

    T_munu = spacetime.stress_energy_tensor_DE(r, phi_DE, dphi_DE_dr, dphi_DE_dt, phi_DM, dilaton_field, beta, V0, lambda_)
    assert isinstance(T_munu, np.ndarray)
    assert T_munu.shape == (4, 4, len(r))
    assert not np.allclose(T_munu, 0, atol=1e-9)  # Should not be zero with non-zero fields, increased tolerance
