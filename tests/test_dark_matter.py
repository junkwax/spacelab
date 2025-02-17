# tests/test_dark_matter.py (CORRECTED)
import pytest
import numpy as np
from src.models.dark_matter import DarkMatter

@pytest.fixture
def dark_matter():
    # Update to use coupling_dilaton and coupling_curvature
    return DarkMatter(mass=1e-22, coupling_dilaton=1e-10, coupling_curvature=1e-5)

def test_density_profile_valid_input(dark_matter):
    # Test with valid scalar input
    r = 1.0
    dilaton_field = 1.0  # Example value
    density = dark_matter.density_profile(r, dilaton_field)
    assert np.isclose(density, 1e-10 * np.exp(-1e-22))

def test_field_equation_with_placeholders(dark_matter):
    y = (0.1, 0.01) # Example phi, dphi_dr
    r = np.array([3.0, 4.0, 5.0])
    dilaton_field = np.array([0.0, 0.0, 0.0])
    graviphoton_field = np.array([0.0, 0.0, 0.0])
    phi_DE = np.array([0.0, 0.0, 0.0])

    dphi_dr, ddphi_dr2 = dark_matter.field_equation(y, r, dilaton_field, graviphoton_field, phi_DE)
    assert isinstance(dphi_dr, np.ndarray)
    assert isinstance(ddphi_dr2, np.ndarray)
    assert dphi_dr.shape == r.shape #Corrected
    assert ddphi_dr2.shape == r.shape

    # Test with a scalar 'r' value
    r_scalar = 3.0
    dphi_dr_scalar, ddphi_dr2_scalar = dark_matter.field_equation((0.1, 0.01), r_scalar, 0.0, 0.0, 0.0)
    assert isinstance(dphi_dr_scalar, np.ndarray)  # Should now be a NumPy array
    assert isinstance(ddphi_dr2_scalar, np.ndarray)  # Should now be a NumPy array
    assert dphi_dr_scalar.shape == (1,) # Check for correct shape
    assert ddphi_dr2_scalar.shape == (1,)
