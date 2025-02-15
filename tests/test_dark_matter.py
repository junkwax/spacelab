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

    # Test with valid array input
    r = np.array()
    dilaton_field = np.array()  # Example values
    density = dark_matter.density_profile(r, dilaton_field)
    expected_density = 1e-10 * np.exp(-1e-22 * r) / r**2
    assert np.allclose(density, expected_density)

def test_density_profile_scalar_input(dark_matter):
    # Test with scalar input for r and dilaton_field
    r = 1.0
    dilaton_field = 1.0  # Example value
    density = dark_matter.density_profile(r, dilaton_field)
    assert np.isclose(density, 1e-10 * np.exp(-1e-22))

def test_density_profile_negative_radius(dark_matter):
    # Test with negative radius (should raise ValueError)
    r = -1.0
    dilaton_field = 1.0  # Example value
    with pytest.raises(ValueError):
        dark_matter.density_profile(r, dilaton_field)

    # Test with array containing negative radius (should raise ValueError)
    r = np.array([1.0, -2.0])
    dilaton_field = np.array()  # Example values
    with pytest.raises(ValueError):
        dark_matter.density_profile(r, dilaton_field)

def test_invalid_mass():
    with pytest.raises(ValueError):
        # Update to use coupling_dilaton and coupling_curvature
        DarkMatter(mass=-1e-22, coupling_dilaton=1e-10, coupling_curvature=1e-5)

def test_invalid_coupling():
    with pytest.raises(ValueError):
        # Update to use coupling_dilaton and coupling_curvature
        DarkMatter(mass=1e-22, coupling_dilaton=-1e-10, coupling_curvature=1e-5)