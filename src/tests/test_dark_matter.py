import pytest
import numpy as np
from src.models.dark_matter import DarkMatter


@pytest.fixture
def dark_matter():
    return DarkMatter(mass=1e-22, coupling_dilaton=1e-10, coupling_curvature=1e-5)


def test_density_profile_valid_input(dark_matter):
    r = 1.0
    dilaton_field = 1.0
    density = dark_matter.density_profile(r, dilaton_field)
    assert np.isclose(density, 1e-10 * np.exp(-1e-22))


def test_density_profile_array(dark_matter):
    r = np.array([1.0, 2.0, 3.0])
    dilaton_field = 1.0
    density = dark_matter.density_profile(r, dilaton_field)
    assert density.shape == (3,)
    assert np.all(density > 0)


def test_density_profile_rejects_zero_radius(dark_matter):
    with pytest.raises(ValueError, match="positive"):
        dark_matter.density_profile(0.0, 1.0)


def test_invalid_mass():
    with pytest.raises(ValueError):
        DarkMatter(mass=-1e-22, coupling_dilaton=1e-10, coupling_curvature=1e-5)


def test_field_equation_returns_plain_floats(dark_matter):
    result = dark_matter.field_equation((0.1, 0.0), 1.0, 1.0, 0.0)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
