import pytest
import numpy as np
from src.observations.gamma_ray import AnnihilationSpectrum, GammaRayFlux, JFactor


def test_spectrum_bb_physical():
    """bb-bar spectrum should be positive for E < m_DM and zero for E > m_DM."""
    spec = AnnihilationSpectrum(m_DM=100.0, channel="bb")
    E_valid = np.array([1.0, 10.0, 50.0])
    E_invalid = np.array([100.0, 200.0])

    assert np.all(spec.dNdE(E_valid) > 0)
    assert np.all(spec.dNdE(E_invalid) == 0)


def test_spectrum_channels():
    """All supported channels should produce non-negative spectra."""
    for channel in ["bb", "tautau", "ww"]:
        spec = AnnihilationSpectrum(m_DM=100.0, channel=channel)
        E = np.geomspace(0.1, 90.0, 50)
        assert np.all(spec.dNdE(E) >= 0)


def test_spectrum_unknown_channel():
    spec = AnnihilationSpectrum(m_DM=100.0, channel="unknown")
    with pytest.raises(ValueError, match="Unknown channel"):
        spec.dNdE(10.0)


def test_gamma_ray_flux_positive():
    """Differential flux should be positive at valid energies."""
    spec = AnnihilationSpectrum(m_DM=100.0, channel="bb")
    flux = GammaRayFlux(
        sigma_v=3e-26,
        m_DM=100.0,
        spectrum=spec,
        j_factor=1e20,
    )
    E = np.array([1.0, 10.0, 50.0])
    dPhi = flux.differential_flux(E)
    assert np.all(dPhi > 0)


def test_integrated_flux_positive():
    spec = AnnihilationSpectrum(m_DM=100.0, channel="bb")
    flux = GammaRayFlux(sigma_v=3e-26, m_DM=100.0, spectrum=spec, j_factor=1e20)
    phi = flux.integrated_flux(1.0, 50.0)
    assert phi > 0


def test_upper_limit_sigma_v():
    """Upper limit on sigma_v should scale linearly with flux limit."""
    spec = AnnihilationSpectrum(m_DM=100.0, channel="bb")
    flux = GammaRayFlux(sigma_v=3e-26, m_DM=100.0, spectrum=spec, j_factor=1e20)

    sv1 = flux.upper_limit_sigma_v(1e-10, 1.0, 50.0)
    sv2 = flux.upper_limit_sigma_v(2e-10, 1.0, 50.0)
    assert np.isclose(sv2 / sv1, 2.0, rtol=1e-3)


def test_j_factor_with_nfw():
    """J-factor should be finite and positive for an NFW-like profile."""
    from src.observations.rotation_curves import NFWProfile

    nfw = NFWProfile(rho_s=1e7, r_s=10.0)

    jf = JFactor(density_func=nfw.density, distance=780.0)  # ~M31 distance
    J = jf.compute(theta_max=np.radians(0.5), l_max=50.0)
    assert J > 0
    assert np.isfinite(J)
