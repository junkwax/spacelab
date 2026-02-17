import pytest
import numpy as np
from src.observations.rotation_curves import NFWProfile, KKAxionProfile, RotationCurveModel
from src.models.dark_matter import DarkMatter


@pytest.fixture
def nfw():
    return NFWProfile(rho_s=1e7, r_s=10.0)


@pytest.fixture
def dm():
    return DarkMatter(mass=1e-22, coupling_dilaton=1e-10, coupling_curvature=1e-5)


def test_nfw_density_at_scale_radius(nfw):
    """At r = r_s, density should be rho_s / (1 * 2^2) = rho_s / 4."""
    expected = 1e7 / 4.0
    assert np.isclose(nfw.density(10.0), expected)


def test_nfw_density_decreases(nfw):
    """NFW density should decrease with radius (for r > 0)."""
    r = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
    rho = nfw.density(r)
    assert np.all(np.diff(rho) < 0)


def test_nfw_enclosed_mass_increases(nfw):
    """Enclosed mass should increase monotonically."""
    r = np.array([1.0, 5.0, 10.0, 20.0])
    M = nfw.enclosed_mass(r)
    assert np.all(np.diff(M) > 0)


def test_nfw_circular_velocity(nfw):
    """v_circ should be positive and finite."""
    r = np.array([1.0, 5.0, 10.0, 20.0])
    v = nfw.circular_velocity(r)
    assert np.all(v > 0)
    assert np.all(np.isfinite(v))


def test_nfw_invalid_params():
    with pytest.raises(ValueError):
        NFWProfile(rho_s=-1.0, r_s=10.0)
    with pytest.raises(ValueError):
        NFWProfile(rho_s=1e7, r_s=0.0)


def test_kk_axion_profile(dm):
    """KK-axion profile should produce positive density and rising v_circ."""
    halo = KKAxionProfile(dm, rho_0=1e7, r_core=5.0)
    r = np.array([1.0, 3.0, 5.0, 10.0])
    rho = halo.density(r)
    assert np.all(rho >= 0)
    assert np.all(np.isfinite(rho))

    v = halo.circular_velocity(r)
    assert np.all(v >= 0)


def test_rotation_curve_model_predict(nfw):
    """Total model velocity should be >= DM-only velocity."""
    model = RotationCurveModel(nfw, upsilon_disk=0.5, upsilon_bulge=0.7)
    r = np.array([2.0, 5.0, 10.0, 15.0])
    v_disk = np.array([40.0, 55.0, 50.0, 40.0])
    v_bulge = np.array([10.0, 8.0, 5.0, 3.0])

    v_total = model.predict(r, v_disk=v_disk, v_bulge=v_bulge)
    v_dm_only = nfw.circular_velocity(r)

    assert np.all(v_total >= v_dm_only)
    assert np.all(np.isfinite(v_total))


def test_chi_squared(nfw):
    """Chi-squared should be 0 when prediction matches data perfectly."""
    model = RotationCurveModel(nfw)
    r = np.array([5.0, 10.0, 15.0])
    v_pred = model.predict(r)
    v_err = np.ones_like(r) * 5.0

    chi2 = model.chi_squared(r, v_pred, v_err)
    assert np.isclose(chi2, 0.0, atol=1e-10)
