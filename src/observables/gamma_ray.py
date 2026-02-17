"""
Gamma-ray flux from dark matter annihilation.

Computes:
  - J-factor:  J = ∫ ρ²(r) dl  (line-of-sight integral through a halo)
  - Differential flux:  dΦ/dE = ⟨σv⟩/(8π m²) × J × dN/dE
  - Integrated flux above threshold energy

Targets: dwarf spheroidal galaxies, galactic center, galaxy clusters.
"""

import numpy as np
from scipy.integrate import quad, dblquad
from typing import Dict, Optional, Callable, Tuple
import logging

# NumPy 2.0 renamed trapz → trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz

logger = logging.getLogger(__name__)

KPC_TO_CM = 3.0857e21   # cm per kpc
GEV_TO_ERG = 1.602e-3   # erg per GeV


# ---------------------------------------------------------------------------
# J-factor computation
# ---------------------------------------------------------------------------

def j_factor_los(
    density_fn: Callable,
    d_target: float,
    r_max: float,
    theta_max: float,
    n_theta: int = 50,
) -> float:
    """Compute the J-factor by integrating ρ² along lines of sight.

    J = ∫∫ ρ²(r(l, θ)) l dl sin(θ) dθ × 2π

    where r(l, θ) = sqrt(l² + d² - 2 l d cos(θ)), d = distance to target.

    Args:
        density_fn: ρ(r) in [mass/length³], takes radius in same units as d_target.
        d_target: Distance to the target [kpc].
        r_max: Maximum halo radius for integration [kpc].
        theta_max: Angular integration radius [radians].
        n_theta: Number of angular bins.

    Returns:
        J-factor in [mass² / length⁵] (same unit system as density_fn).
    """
    d = d_target

    def integrand_l(l, theta):
        r = np.sqrt(l**2 + d**2 - 2 * l * d * np.cos(theta))
        r = max(r, 1e-10)
        if r > r_max:
            return 0.0
        rho = density_fn(r)
        return rho**2

    # l_max from geometry: max line-of-sight distance through sphere of r_max
    l_max = d * np.cos(theta_max) + np.sqrt(r_max**2 - (d * np.sin(theta_max))**2 + 1e-30)
    l_max = max(l_max, 0.1)

    J = 0.0
    thetas = np.linspace(0, theta_max, n_theta)

    for theta in thetas:
        if theta == 0:
            continue
        val, _ = quad(integrand_l, 0, l_max, args=(theta,), limit=200, epsrel=1e-4)
        J += val * np.sin(theta) * (thetas[1] - thetas[0])

    J *= 2.0 * np.pi  # azimuthal symmetry
    return J


def j_factor_spherical(
    density_fn: Callable,
    r_min: float = 1e-3,
    r_max: float = 100.0,
) -> float:
    """Simplified J-factor for a spherical halo (volume integral).

    J_vol = ∫ ρ²(r) 4π r² dr

    This is the volume-averaged ⟨ρ²⟩ × Volume, useful for
    order-of-magnitude estimates and dwarf spheroidals.
    """
    integrand = lambda r: 4.0 * np.pi * r**2 * density_fn(r)**2
    J, _ = quad(integrand, r_min, r_max, limit=200, epsrel=1e-6)
    return J


# ---------------------------------------------------------------------------
# Annihilation spectra (parametric approximations)
# ---------------------------------------------------------------------------

def bb_spectrum(E: np.ndarray, m_dm: float) -> np.ndarray:
    """Approximate dN/dE for DM DM → bb̄ → γ (Cirelli et al. parametrization).

    Simple log-parabola fit valid for m_dm > 10 GeV.

    Args:
        E: Photon energies [GeV].
        m_dm: Dark matter mass [GeV].

    Returns:
        dN/dE [GeV⁻¹] per annihilation.
    """
    x = E / m_dm
    # Mask out kinematically forbidden
    result = np.zeros_like(E)
    valid = (x > 1e-6) & (x < 1.0)

    # Log-parabola approximation
    lnx = np.log(x[valid])
    result[valid] = np.exp(-7.0 * x[valid]) * (-lnx)**1.5 / (E[valid])
    # Normalize: total number of photons ≈ 25 for bb̄ at ~100 GeV
    norm = 25.0 / _trapz(result[valid], E[valid]) if np.any(valid) else 1.0
    result[valid] *= norm
    return result


def tau_spectrum(E: np.ndarray, m_dm: float) -> np.ndarray:
    """Approximate dN/dE for DM DM → τ⁺τ⁻ → γ.

    Harder spectrum than bb̄.
    """
    x = E / m_dm
    result = np.zeros_like(E)
    valid = (x > 1e-6) & (x < 1.0)

    lnx = np.log(x[valid])
    result[valid] = np.exp(-4.0 * x[valid]) * (-lnx)**1.0 / E[valid]
    norm = 10.0 / _trapz(result[valid], E[valid]) if np.any(valid) else 1.0
    result[valid] *= norm
    return result


CHANNEL_SPECTRA = {
    "bb": bb_spectrum,
    "tau": tau_spectrum,
}


# ---------------------------------------------------------------------------
# Flux calculation
# ---------------------------------------------------------------------------

class GammaRayFlux:
    """Dark matter annihilation gamma-ray flux calculator.

    Args:
        m_dm: Dark matter particle mass [GeV].
        sigma_v: Thermally-averaged annihilation cross section [cm³/s].
        channel: Annihilation channel ('bb' or 'tau').
    """

    def __init__(
        self,
        m_dm: float = 100.0,
        sigma_v: float = 3e-26,
        channel: str = "bb",
    ):
        self.m_dm = m_dm
        self.sigma_v = sigma_v
        self.channel = channel

        if channel not in CHANNEL_SPECTRA:
            raise ValueError(f"Unknown channel '{channel}'. Options: {list(CHANNEL_SPECTRA.keys())}")

    def spectrum(self, E: np.ndarray) -> np.ndarray:
        """dN/dE [GeV⁻¹] for the chosen channel."""
        return CHANNEL_SPECTRA[self.channel](E, self.m_dm)

    def differential_flux(
        self,
        E: np.ndarray,
        J_factor: float,
    ) -> np.ndarray:
        """Compute dΦ/dE [cm⁻² s⁻¹ GeV⁻¹].

        dΦ/dE = ⟨σv⟩ / (8π m²) × J × dN/dE

        Args:
            E: Photon energies [GeV].
            J_factor: J-factor in [GeV² cm⁻⁵] or consistent units.

        Returns:
            Differential flux at each energy.
        """
        prefactor = self.sigma_v / (8.0 * np.pi * self.m_dm**2)
        return prefactor * J_factor * self.spectrum(E)

    def integrated_flux(
        self,
        E_min: float,
        E_max: float,
        J_factor: float,
        n_points: int = 200,
    ) -> float:
        """Integrate dΦ/dE from E_min to E_max [GeV].

        Returns:
            Total flux [cm⁻² s⁻¹].
        """
        E = np.geomspace(E_min, E_max, n_points)
        dPhi = self.differential_flux(E, J_factor)
        return _trapz(dPhi, E)

    def upper_limit_sigma_v(
        self,
        flux_limit: float,
        E_min: float,
        E_max: float,
        J_factor: float,
    ) -> float:
        """Given an observed flux upper limit, derive ⟨σv⟩ upper limit.

        Inverts the flux equation at the given J-factor.
        """
        # Compute flux for sigma_v = 1 cm³/s
        saved = self.sigma_v
        self.sigma_v = 1.0
        flux_unit = self.integrated_flux(E_min, E_max, J_factor)
        self.sigma_v = saved

        if flux_unit <= 0:
            return np.inf
        return flux_limit / flux_unit


# ---------------------------------------------------------------------------
# Convenience: Fermi-LAT dwarf spheroidal comparison
# ---------------------------------------------------------------------------

# Approximate J-factors for well-known targets [log10(J / GeV² cm⁻⁵)]
DWARF_J_FACTORS = {
    "draco":        18.8,
    "sculptor":     18.6,
    "ursa_minor":   18.8,
    "carina":       17.9,
    "fornax":       18.2,
    "leo_I":        17.8,
    "sextans":      17.5,
    "segue_1":      19.5,
    "reticulum_II": 18.9,
    "tucana_II":    18.8,
}


def fermi_lat_comparison(
    flux_calc: GammaRayFlux,
    density_fn: Optional[Callable] = None,
    E_min: float = 0.5,
    E_max: float = 500.0,
) -> Dict[str, Dict[str, float]]:
    """Compare predicted fluxes against typical Fermi-LAT sensitivity.

    Uses tabulated J-factors for classical dwarf spheroidals.

    Args:
        flux_calc: GammaRayFlux instance.
        density_fn: If provided, compute J-factor from this profile instead.
        E_min, E_max: Energy range [GeV].

    Returns:
        Dict per target with 'J_factor', 'predicted_flux', 'detectable'.
    """
    # Approximate Fermi-LAT 10-year sensitivity [cm⁻² s⁻¹]
    FERMI_SENSITIVITY = 2e-12

    results = {}
    for name, log10_J in DWARF_J_FACTORS.items():
        J = 10**log10_J

        if density_fn is not None:
            J_computed = j_factor_spherical(density_fn)
            # Convert units if needed (this is approximate)
            J = J_computed

        flux = flux_calc.integrated_flux(E_min, E_max, J)

        results[name] = {
            "log10_J": log10_J,
            "J_factor": J,
            "predicted_flux": flux,
            "fermi_sensitivity": FERMI_SENSITIVITY,
            "detectable": flux > FERMI_SENSITIVITY,
        }

    return results
