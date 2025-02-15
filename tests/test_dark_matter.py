import pytest
import numpy as np
from src.models.dark_matter import DarkMatter

@pytest.fixture
def dark_matter():
    return DarkMatter(mass=1e-22, coupling=1e-10)

def test_density_profile_valid_input(dark_matter):
    r = np.array([1.0, 10.0, 100.0])
    density = dark_matter.density_profile(r)
    expected = dark_matter.coupling * np.exp(-dark_matter.mass * r) / r**2
    assert np.allclose(density, expected)

def test_density_profile_scalar_input(dark_matter):
    r = 10.0
    density = dark_matter.density_profile(r)
    expected = dark_matter.coupling * np.exp(-dark_matter.mass * r) / r**2
    assert np.isclose(density, expected)

def test_density_profile_negative_radius(dark_matter):
    with pytest.raises(ValueError):
        dark_matter.density_profile(-1.0)

def test_invalid_mass():
    with pytest.raises(ValueError):
        DarkMatter(mass=-1e-22, coupling=1e-10)

def test_invalid_coupling():
    with pytest.raises(ValueError):
        DarkMatter(mass=1e-22, coupling=-1e-10)