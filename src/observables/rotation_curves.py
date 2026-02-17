"""
Galaxy rotation curve predictions and SPARC data integration.

Computes circular velocity v(r) = sqrt(GM(<r)/r) from:
  - Dark matter halo  (NFW or SpaceLab axion profile)
  - Baryonic components read directly from SPARC data columns
    (Vgas, Vdisk, Vbul already decomposed by Lelli+2016)

SPARC file format (Rotmod_LTG):
  Col 0: Rad    [kpc]     — Galactocentric radius
  Col 1: Vobs   [km/s]    — Observed rotation velocity
  Col 2: errV   [km/s]    — Uncertainty on Vobs
  Col 3: Vgas   [km/s]    — Gas contribution (from HI surface density)
  Col 4: Vdisk  [km/s]    — Stellar disk contribution (Υ★=1 M☉/L☉)
  Col 5: Vbul   [km/s]    — Bulge contribution (Υ★=1 M☉/L☉)
  Col 6: SBdisk [L☉/pc²]  — Disk surface brightness at 3.6 μm
  Col 7: SBbul  [L☉/pc²]  — Bulge surface brightness at 3.6 μm

Key: SPARC provides Vdisk and Vbul for Υ★=1.  To get the physical
contribution, multiply by sqrt(Υ★):  Vdisk_phys = Vdisk × sqrt(Υ★_disk).
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import i0, i1, k0, k1
from typing import Dict, Optional, List, Tuple
import logging
import os
import glob

logger = logging.getLogger(__name__)

G_N = 6.67430e-11       # m³ kg⁻¹ s⁻²
KPC_TO_M = 3.0857e19    # meters per kpc
MSUN_KG = 1.98847e30    # kg per solar mass


# ===================================================================
# SPARC data loading
# ===================================================================

class SPARCGalaxy:
    """Container for a single SPARC galaxy rotation curve.

    Attributes:
        name: Galaxy name (e.g. 'NGC2403').
        r_kpc: Radii [kpc].
        v_obs: Observed rotation velocity [km/s].
        v_err: Velocity uncertainty [km/s].
        v_gas: Gas contribution [km/s].
        v_disk: Stellar disk contribution at Υ★=1 [km/s].
        v_bul: Bulge contribution at Υ★=1 [km/s].
        sb_disk: Disk surface brightness [L☉/pc²].
        sb_bul: Bulge surface brightness [L☉/pc²].
    """

    def __init__(self, name: str, data: np.ndarray):
        self.name = name
        self.r_kpc = data[:, 0]
        self.v_obs = data[:, 1]
        self.v_err = data[:, 2]
        self.v_gas = data[:, 3] if data.shape[1] > 3 else np.zeros_like(self.r_kpc)
        self.v_disk = data[:, 4] if data.shape[1] > 4 else np.zeros_like(self.r_kpc)
        self.v_bul = data[:, 5] if data.shape[1] > 5 else np.zeros_like(self.r_kpc)
        self.sb_disk = data[:, 6] if data.shape[1] > 6 else np.zeros_like(self.r_kpc)
        self.sb_bul = data[:, 7] if data.shape[1] > 7 else np.zeros_like(self.r_kpc)

    @property
    def n_points(self) -> int:
        return len(self.r_kpc)

    @property
    def r_max(self) -> float:
        return self.r_kpc[-1]

    @property
    def v_flat(self) -> float:
        """Estimate the flat rotation velocity (mean of outer 1/3)."""
        n = max(1, len(self.v_obs) // 3)
        return float(np.mean(self.v_obs[-n:]))

    def baryonic_velocity(self, upsilon_disk: float = 0.5, upsilon_bul: float = 0.7) -> np.ndarray:
        """Total baryonic velocity for given mass-to-light ratios.

        V_bar² = V_gas² + Υ★_disk × V_disk² + Υ★_bul × V_bul²
        """
        v_bar_sq = (
            self.v_gas**2
            + upsilon_disk * self.v_disk**2
            + upsilon_bul * self.v_bul**2
        )
        return np.sqrt(np.maximum(v_bar_sq, 0.0))

    def dm_velocity_needed(self, upsilon_disk: float = 0.5, upsilon_bul: float = 0.7) -> np.ndarray:
        """Dark matter velocity needed: V_DM² = V_obs² - V_bar²."""
        v_bar = self.baryonic_velocity(upsilon_disk, upsilon_bul)
        v_dm_sq = self.v_obs**2 - v_bar**2
        return np.sqrt(np.maximum(v_dm_sq, 0.0))

    def as_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dict format expected by the fitting pipeline."""
        return {
            "r_kpc": self.r_kpc,
            "v_obs_kms": self.v_obs,
            "v_err_kms": self.v_err,
            "v_gas_kms": self.v_gas,
            "v_disk_kms": self.v_disk,
            "v_bul_kms": self.v_bul,
        }


def load_sparc_galaxy(filepath: str) -> SPARCGalaxy:
    """Load a single SPARC rotation curve file.

    Args:
        filepath: Path to a *_rotmod.dat file.

    Returns:
        SPARCGalaxy instance.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"SPARC file not found: {filepath}")

    name = os.path.basename(filepath).replace("_rotmod.dat", "")
    data = np.loadtxt(filepath, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    logger.info(f"Loaded {name}: {data.shape[0]} points, r=[{data[0,0]:.2f}, {data[-1,0]:.2f}] kpc")
    return SPARCGalaxy(name, data)


def load_sparc_catalog(data_dir: str = "data/sparc") -> Dict[str, SPARCGalaxy]:
    """Load all SPARC galaxies from a directory.

    Expects files named *_rotmod.dat in the SPARC format.

    Returns:
        Dict mapping galaxy name -> SPARCGalaxy.
    """
    pattern = os.path.join(data_dir, "*_rotmod.dat")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No SPARC files found in {data_dir}. "
            f"Run `python scripts/download_sparc.py` first."
        )

    catalog = {}
    for f in files:
        gal = load_sparc_galaxy(f)
        catalog[gal.name] = gal

    logger.info(f"Loaded {len(catalog)} galaxies from {data_dir}")
    return catalog


# ===================================================================
# Dark matter halo profiles
# ===================================================================

def nfw_velocity(r_kpc: np.ndarray, rho_s: float, r_s: float) -> np.ndarray:
    """NFW halo circular velocity [km/s].

    Args:
        r_kpc: Radii [kpc].
        rho_s: Scale density [M☉/kpc³].
        r_s: Scale radius [kpc].

    Returns:
        V_halo [km/s].
    """
    r = np.asarray(r_kpc, dtype=float)
    x = r / r_s

    # M(<r) = 4π ρ_s r_s³ [ln(1+x) - x/(1+x)]
    M_enc = 4.0 * np.pi * rho_s * r_s**3 * (np.log(1.0 + x) - x / (1.0 + x))

    # V² = G M / r, convert to km/s
    r_m = r * KPC_TO_M
    M_kg = M_enc * MSUN_KG
    v_sq = G_N * M_kg / r_m  # m²/s²
    return np.sqrt(np.maximum(v_sq, 0.0)) / 1e3  # → km/s


def nfw_from_c200_M200(
    r_kpc: np.ndarray,
    log10_M200: float,
    c200: float,
    H0: float = 67.4,
) -> np.ndarray:
    """NFW velocity from virial mass and concentration.

    Parameterizes the halo by (M200, c200) instead of (ρ_s, r_s).

    Args:
        r_kpc: Radii [kpc].
        log10_M200: log10 of virial mass [M☉].
        c200: Concentration parameter.
        H0: Hubble constant [km/s/Mpc].

    Returns:
        V_halo [km/s].
    """
    M200 = 10**log10_M200  # M☉

    # ρ_crit = 3H²/(8πG) in M☉/kpc³
    H_si = H0 * 1e3 / 3.0857e22  # s⁻¹
    rho_crit = 3.0 * H_si**2 / (8.0 * np.pi * G_N)  # kg/m³
    rho_crit_astro = rho_crit / MSUN_KG * KPC_TO_M**3  # M☉/kpc³

    # r200 from M200 = (4/3)π × 200 ρ_crit × r200³
    r200 = (3.0 * M200 / (4.0 * np.pi * 200 * rho_crit_astro))**(1.0 / 3.0)  # kpc

    r_s = r200 / c200
    rho_s = 200.0 * rho_crit_astro * c200**3 / (3.0 * (np.log(1 + c200) - c200 / (1 + c200)))

    return nfw_velocity(r_kpc, rho_s, r_s)


def axion_velocity(
    r_kpc: np.ndarray,
    log10_rho0: float,
    r_c: float,
    mass_param: float = 1e-22,
) -> np.ndarray:
    """SpaceLab axion-like DM halo velocity [km/s].

    Uses a soliton-core + exponential envelope profile:
        ρ(r) = ρ₀ / (1 + (r/r_c)²)^β × exp(-m_a × r)

    This is physically motivated by the axion field equation
    (BEC ground state produces a soliton core).

    Args:
        r_kpc: Radii [kpc].
        log10_rho0: log10 central density [M☉/kpc³].
        r_c: Core radius [kpc].
        mass_param: Axion mass parameter (controls envelope decay).

    Returns:
        V_halo [km/s].
    """
    rho0 = 10**log10_rho0
    beta = 4.0  # soliton index (from Schive+2014: ρ ~ 1/(1 + 0.091(r/rc)²)⁸)

    r = np.asarray(r_kpc, dtype=float)
    result = np.zeros_like(r)

    for i, r_max in enumerate(r):
        if r_max <= 0:
            continue

        def density(rp):
            return rho0 / (1.0 + (rp / r_c)**2)**beta * np.exp(-mass_param * rp)

        def integrand(rp):
            return 4.0 * np.pi * rp**2 * density(rp)

        M_enc, _ = quad(integrand, 0, r_max, limit=200)
        r_m = r_max * KPC_TO_M
        M_kg = M_enc * MSUN_KG
        result[i] = np.sqrt(max(G_N * M_kg / r_m, 0.0)) / 1e3

    return result


# ===================================================================
# Model function builders (for fitting)
# ===================================================================

def build_nfw_model(galaxy: SPARCGalaxy):
    """Build an NFW model function for a SPARC galaxy.

    Fits 3 parameters: log10(rho_s), r_s, Υ★_disk.
    Gas is fixed from the data, bulge uses Υ★=0.7.

    Returns:
        (model_fn, param_names, priors)
    """
    param_names = ["log10_rho_s", "r_s_kpc", "upsilon_disk"]
    priors = {
        "log10_rho_s": (5.0, 10.0),
        "r_s_kpc": (0.5, 100.0),
        "upsilon_disk": (0.1, 1.5),
    }

    def model_fn(r_kpc_arr, log10_rho_s, r_s_kpc, upsilon_disk):
        """Returns total V [km/s] at radii r_kpc_arr."""
        # DM halo
        rho_s = 10**log10_rho_s
        v_halo = nfw_velocity(r_kpc_arr, rho_s, r_s_kpc)

        # Baryons: interpolate SPARC columns to requested radii
        v_gas = np.interp(r_kpc_arr, galaxy.r_kpc, galaxy.v_gas)
        v_disk = np.interp(r_kpc_arr, galaxy.r_kpc, galaxy.v_disk)
        v_bul = np.interp(r_kpc_arr, galaxy.r_kpc, galaxy.v_bul)

        v_bar_sq = v_gas**2 + upsilon_disk * v_disk**2 + 0.7 * v_bul**2
        v_total = np.sqrt(v_halo**2 + np.maximum(v_bar_sq, 0.0))
        return v_total

    return model_fn, param_names, priors


def build_axion_model(galaxy: SPARCGalaxy):
    """Build an axion soliton+envelope model for a SPARC galaxy.

    Fits 3 parameters: log10(rho0), r_c, Υ★_disk.

    Returns:
        (model_fn, param_names, priors)
    """
    param_names = ["log10_rho0", "r_c_kpc", "upsilon_disk"]
    priors = {
        "log10_rho0": (5.0, 12.0),
        "r_c_kpc": (0.1, 20.0),
        "upsilon_disk": (0.1, 1.5),
    }

    def model_fn(r_kpc_arr, log10_rho0, r_c_kpc, upsilon_disk):
        v_halo = axion_velocity(r_kpc_arr, log10_rho0, r_c_kpc)

        v_gas = np.interp(r_kpc_arr, galaxy.r_kpc, galaxy.v_gas)
        v_disk = np.interp(r_kpc_arr, galaxy.r_kpc, galaxy.v_disk)
        v_bul = np.interp(r_kpc_arr, galaxy.r_kpc, galaxy.v_bul)

        v_bar_sq = v_gas**2 + upsilon_disk * v_disk**2 + 0.7 * v_bul**2
        v_total = np.sqrt(v_halo**2 + np.maximum(v_bar_sq, 0.0))
        return v_total

    return model_fn, param_names, priors


def build_nfw_c200_model(galaxy: SPARCGalaxy):
    """NFW parameterized by (M200, c200, Υ★) for cosmological comparison.

    Returns:
        (model_fn, param_names, priors)
    """
    param_names = ["log10_M200", "c200", "upsilon_disk"]
    priors = {
        "log10_M200": (8.0, 14.0),
        "c200": (2.0, 50.0),
        "upsilon_disk": (0.1, 1.5),
    }

    def model_fn(r_kpc_arr, log10_M200, c200, upsilon_disk):
        v_halo = nfw_from_c200_M200(r_kpc_arr, log10_M200, c200)

        v_gas = np.interp(r_kpc_arr, galaxy.r_kpc, galaxy.v_gas)
        v_disk = np.interp(r_kpc_arr, galaxy.r_kpc, galaxy.v_disk)
        v_bul = np.interp(r_kpc_arr, galaxy.r_kpc, galaxy.v_bul)

        v_bar_sq = v_gas**2 + upsilon_disk * v_disk**2 + 0.7 * v_bul**2
        v_total = np.sqrt(v_halo**2 + np.maximum(v_bar_sq, 0.0))
        return v_total

    return model_fn, param_names, priors


# ===================================================================
# Convenience: synthetic data generator (for testing)
# ===================================================================

def generate_synthetic_sparc(
    v_flat: float = 150.0,
    r_max_kpc: float = 30.0,
    n_points: int = 25,
    noise_frac: float = 0.05,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate a simple synthetic rotation curve for pipeline testing."""
    rng = np.random.default_rng(seed)
    r_kpc = np.linspace(0.5, r_max_kpc, n_points)

    # Simple arctan model: V(r) = Vflat × (2/π) arctan(r/r_t)
    r_t = 3.0
    v_true = v_flat * (2.0 / np.pi) * np.arctan(r_kpc / r_t)
    v_err = noise_frac * v_true + 2.0
    v_obs = v_true + rng.normal(0, v_err)

    return {
        "r_kpc": r_kpc,
        "v_obs_kms": v_obs,
        "v_err_kms": v_err,
    }
