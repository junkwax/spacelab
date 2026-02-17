import pytest
import numpy as np
from src.models.cosmology import FRWCosmology
from src.models.dark_energy import QuintessenceField


@pytest.fixture
def cosmo():
    return FRWCosmology(H0=67.4, Omega_m=0.315, Omega_r=9.1e-5)


def test_hubble_today(cosmo):
    """H(a=1) should equal H0 (in Gyr^-1 units)."""
    H_today = cosmo.hubble(1.0)
    assert np.isclose(H_today, cosmo.H0_Gyr, rtol=1e-3)


def test_hubble_increases_at_high_z(cosmo):
    """H should be larger at earlier times (smaller a)."""
    assert cosmo.hubble(0.5) > cosmo.hubble(1.0)
    assert cosmo.hubble(0.1) > cosmo.hubble(0.5)


def test_hubble_vectorized(cosmo):
    a = np.array([0.1, 0.5, 1.0])
    H = cosmo.hubble(a)
    assert H.shape == (3,)
    assert np.all(np.diff(H) < 0)  # H decreases with increasing a


def test_flat_universe(cosmo):
    """Omega_m + Omega_r + Omega_DE should equal 1."""
    assert np.isclose(cosmo.Omega_m + cosmo.Omega_r + cosmo.Omega_DE, 1.0)


def test_comoving_distance_increases(cosmo):
    """Comoving distance should increase with redshift."""
    d1 = cosmo.comoving_distance(0.5)
    d2 = cosmo.comoving_distance(1.0)
    assert d2 > d1 > 0


def test_evolve_quintessence(cosmo):
    """Basic check that quintessence evolution runs and returns sane results."""
    qf = QuintessenceField(V0=1e-47, lambda_=0.1)
    result = cosmo.evolve_quintessence(qf, phi_0=1.0, dphi_dt_0=0.0, n_points=50)

    assert set(result.keys()) == {"a", "t", "phi", "dphi_dt", "H", "w_DE"}
    assert len(result["a"]) == 50
    assert np.all(np.isfinite(result["phi"]))
    assert np.all(np.isfinite(result["w_DE"]))
    # Scale factor should be monotonically increasing
    assert np.all(np.diff(result["a"]) >= 0)
