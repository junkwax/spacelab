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