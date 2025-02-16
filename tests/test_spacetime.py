# tests/test_spacetime.py

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

def test_stress_energy_tensor_DM_zero_fields():
    spacetime = SpacetimeGeometry(mass=1)
    r = np.linspace(3, 10, 10)
    phi_DM = np.zeros_like(r)
    dphi_DM_dr = np.zeros_like(r)
    phi_DE = np.zeros_like(r)
    dilaton_field = np.zeros_like(r)
    beta = 1e-10

    T_munu = spacetime.stress_energy_tensor_DM(r, phi_DM, dphi_DM_dr, phi_DE, dilaton_field, beta)

    assert isinstance(T_munu, np.ndarray)
    assert T_munu.shape == (4, 4, len(r))
    assert np.allclose(T_munu, 0, atol=1e-12)  # Expect zero stress-energy


def test_stress_energy_tensor_DM_simple_case():
    spacetime = SpacetimeGeometry(mass=1)
    r = np.linspace(3, 10, 10)
    phi_DM = np.ones_like(r) * 0.1  # Small constant value
    dphi_DM_dr = np.zeros_like(r)
    phi_DE = np.ones_like(r) * 0.05
    dilaton_field = np.ones_like(r) * 0.01
    beta = 1e-10

    T_munu = spacetime.stress_energy_tensor_DM(r, phi_DM, dphi_DM_dr, phi_DE, dilaton_field, beta)

    assert isinstance(T_munu, np.ndarray)
    assert T_munu.shape == (4, 4, len(r))
    assert not np.allclose(T_munu, 0, atol=1e-12) #Should not all be zero
    #Add more specific checks if possible, based on expected behavior

def test_stress_energy_tensor_DE_zero_fields():
    spacetime = SpacetimeGeometry(mass=1)
    r = np.linspace(3, 10, 10)
    phi_DE = np.zeros_like(r)
    dphi_DE_dr = np.zeros_like(r)
    dphi_DE_dt = np.zeros_like(r)
    phi_DM = np.zeros_like(r)
    dilaton_field = np.zeros_like(r)
    beta = 1e-10
    V0 = 1e-47
    lambda_ = 0.1

    T_munu = spacetime.stress_energy_tensor_DE(r, phi_DE, dphi_DE_dr, dphi_DE_dt, phi_DM, dilaton_field, beta, V0, lambda_)

    assert isinstance(T_munu, np.ndarray)
    assert T_munu.shape == (4, 4, len(r))
    assert np.allclose(T_munu, 0, atol=1e-12)  # Expect zero stress-energy

def test_stress_energy_tensor_DE_simple_case():
    spacetime = SpacetimeGeometry(mass=1)
    r = np.linspace(3, 10, 10)
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
    assert not np.allclose(T_munu, 0, atol=1e-12) # Should not all be zero
    #Add more specific checks based on expected behavior
