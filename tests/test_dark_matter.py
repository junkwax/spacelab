import pytest
import numpy as np
from src.models.dark_matter import DarkMatter

@pytest.fixture
def dark_matter():
    # Initialize DarkMatter with example parameters
    return DarkMatter(mass=1e-22, coupling_dilaton=1e-10, coupling_curvature=1e-5)

def test_density_profile_valid_input(dark_matter):
    # Test with valid scalar input
    r = 1.0
    dilaton_field = 1.0  # Example value
    density = dark_matter.density_profile(r, dilaton_field)
    assert np.isclose(density, 1e-10 * np.exp(-1e-22))

def test_field_equation_with_placeholders(dark_matter):
    # Test with array inputs
    y = (np.array([0.1, 0.1, 0.1]), np.array([0.01, 0.01, 0.01]))  # Example phi, dphi_dr
    r = np.array([3.0e9, 4.0e9, 5.0e9])
    dilaton_field = np.array([0.0, 0.0, 0.0])
    graviphoton_field = np.array([0.0, 0.0, 0.0])
    phi_DE = np.array([0.0, 0.0, 0.0])

    dphi_dr, ddphi_dr2 = dark_matter.field_equation(y, r, dilaton_field, graviphoton_field, phi_DE)
    assert isinstance(dphi_dr, np.ndarray)
    assert isinstance(ddphi_dr2, np.ndarray)
    assert dphi_dr.shape == r.shape
    assert ddphi_dr2.shape == r.shape

    # Test with a scalar 'r' value
    r_scalar = 3.0e9
    dphi_dr_scalar, ddphi_dr2_scalar = dark_matter.field_equation((np.array([0.1]), np.array([0.01])), r_scalar, 0.0, 0.0, 0.0)
    assert isinstance(dphi_dr_scalar, np.ndarray)
    assert isinstance(ddphi_dr2_scalar, np.ndarray)
    assert dphi_dr_scalar.size == 1  # Check for correct shape
    assert ddphi_dr2_scalar.size == 1

def test_field_equation_derivative_calculation(dark_matter):
    # Test with non-zero, non-constant fields
    y = (np.array([0.1, 0.11, 0.12]), np.array([0.0, 0.0, 0.0]))  # phi_DM, dphi_DM_dr = 0 (initial derivative)
    r = np.array([3.0e9, 4.0e9, 5.0e9])
    dilaton_field = np.array([0.1, 0.2, 0.3])  # Non-constant dilaton
    graviphoton_field = np.array([0.0, 0.0, 0.0])
    phi_DE = np.array([0.2, 0.3, 0.4])  # Non-constant dark energy

    dphi_dr, ddphi_dr2 = dark_matter.field_equation(y, r, dilaton_field, graviphoton_field, phi_DE)

    # Check that dphi_dr has correct values
    assert np.allclose(dphi_dr, np.array([0.0, 0.0, 0.0]))
    # Check that ddphi_dr2 values are not all the same with tighter tolerance
    assert not np.allclose(ddphi_dr2, ddphi_dr2[0], atol=1e-12)
