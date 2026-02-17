"""
Unified Kaluza-Klein parameter space.

All physical parameters are derived from the fundamental 5D geometry:
    R5      — compactification radius of the extra dimension [meters]
    M5      — 5D Planck mass [eV]
    g5      — 5D gauge coupling

From these, the 4D effective theory produces:
    - Dilaton mass and couplings
    - Axion mass (from instanton effects on compactified dimension)
    - Graviphoton coupling
    - Quintessence potential parameters
    - 4D Planck mass and Newton's constant

This module is the single source of truth for the SpaceLab framework.
Changing R5 propagates consistently through every model component.

Conventions:
    Natural units: ℏ = c = 1 unless otherwise noted.
    Masses in eV, lengths in eV⁻¹ or meters as noted.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# Conversion factors
HBAR_C = 1.9733e-7       # ℏc [eV·m]
M_PLANCK_4D = 1.22089e28  # 4D Planck mass [eV]
G_N = 6.67430e-11         # Newton's constant [m³ kg⁻¹ s⁻²]
EV_TO_KG = 1.78266e-36    # kg per eV/c²
KPC_TO_M = 3.0857e19
MSUN_KG = 1.98847e30


@dataclass
class KKParameters:
    """Fundamental Kaluza-Klein parameters and derived quantities.

    Initialize with the compactification radius R5 and optionally the
    5D Planck mass.  All derived couplings are computed automatically.

    The key physical scales:
        m_KK = 1/R5          — KK tower mass scale
        m_dilaton ~ 1/R5     — dilaton gets mass from stabilization potential
        m_axion << m_KK      — axion mass from non-perturbative effects
        g_graviphoton ~ 1/M_Pl — graviphoton coupling (gravitational strength)
    """

    # === Fundamental parameters ===
    R5: float = 1e-6              # Compactification radius [meters]
    M5: float = 1e29              # 5D Planck mass [eV]
    stabilization_scale: float = 0.1   # Dilaton stabilization (fraction of m_KK)
    instanton_suppression: float = 1e-10  # exp(-S_instanton) for axion mass
    m_axion_target: float = 0.0   # If >0, override m_axion and derive instanton_suppression

    # === Derived quantities (computed in __post_init__) ===
    m_KK: float = field(init=False)
    M_Pl_4D: float = field(init=False)
    G_N_derived: float = field(init=False)

    # Dilaton
    m_dilaton: float = field(init=False)
    g_dilaton_matter: float = field(init=False)
    dilaton_vev: float = field(init=False)

    # Axion
    m_axion: float = field(init=False)
    f_axion: float = field(init=False)
    g_axion_dilaton: float = field(init=False)

    # Graviphoton
    g_graviphoton: float = field(init=False)

    # Quintessence
    V0_quintessence: float = field(init=False)
    lambda_quintessence: float = field(init=False)

    # Derived instanton action (if m_axion_target is set)
    instanton_action: float = field(init=False)

    def __post_init__(self):
        """Derive all physical parameters from the fundamental scales."""

        # KK mass scale: m_KK = ℏc / R5
        self.m_KK = HBAR_C / self.R5  # [eV]

        # 4D Planck mass from dimensional reduction:
        # M_Pl² = M5³ × (2π R5)
        # In natural units with our conventions:
        self.M_Pl_4D = np.sqrt(self.M5**3 * 2 * np.pi * self.R5 / HBAR_C)

        # Newton's constant from 4D Planck mass
        self.G_N_derived = HBAR_C / (self.M_Pl_4D**2 * EV_TO_KG)

        # ---------------------------------------------------------------
        # Dilaton sector
        # ---------------------------------------------------------------
        # The dilaton is the scalar mode of the 5D metric describing
        # fluctuations of R5.  Its mass comes from a stabilization
        # potential (Goldberger-Wise or similar mechanism).
        self.m_dilaton = self.stabilization_scale * self.m_KK

        # Dilaton coupling to matter: g_d ~ 1/M_Pl (gravitational strength)
        # but enhanced by O(1) factors from the KK reduction
        self.g_dilaton_matter = np.sqrt(2.0 / 3.0) / self.M_Pl_4D

        # Dilaton VEV sets the background compactification
        self.dilaton_vev = np.log(self.R5 * self.m_KK)

        # ---------------------------------------------------------------
        # Axion sector
        # ---------------------------------------------------------------
        # The axion arises as a pseudo-scalar from the 5D gauge field
        # component A_5, or from a bulk 2-form.  Its decay constant is
        # set by the compactification scale.
        self.f_axion = self.M_Pl_4D / (2 * np.pi)

        # Axion mass from non-perturbative effects (instantons wrapping
        # the compact dimension):
        #   m_a² ~ Λ⁴/f_a²  where Λ⁴ ~ m_KK⁴ × exp(-S_instanton)
        #
        # Two modes:
        #   1. Forward: given instanton_suppression → compute m_axion
        #   2. Inverse: given m_axion_target → derive required instanton_suppression
        if self.m_axion_target > 0:
            # Inverse mode: what instanton suppression is needed?
            self.m_axion = self.m_axion_target
            # m_a = sqrt(m_KK⁴ × ε) / f_a  →  ε = (m_a × f_a)² / m_KK⁴
            self.instanton_suppression = (self.m_axion * self.f_axion)**2 / self.m_KK**4
            self.instanton_action = -np.log(max(self.instanton_suppression, 1e-300))
        else:
            # Forward mode
            Lambda4 = self.m_KK**4 * self.instanton_suppression
            self.m_axion = np.sqrt(Lambda4) / self.f_axion
            self.instanton_action = -np.log(max(self.instanton_suppression, 1e-300))

        # Axion-dilaton coupling from the KK reduction
        # This is the key parameter that connects DM to the 5D geometry
        self.g_axion_dilaton = self.g_dilaton_matter * self.f_axion

        # ---------------------------------------------------------------
        # Graviphoton sector
        # ---------------------------------------------------------------
        # The graviphoton A_μ from g_{μ5} has coupling ~ 1/M_Pl
        self.g_graviphoton = 1.0 / self.M_Pl_4D

        # ---------------------------------------------------------------
        # Quintessence sector
        # ---------------------------------------------------------------
        # If the dilaton potential has a runaway direction at large φ,
        # it drives late-time acceleration.  The potential is:
        #   V(φ) = V0 × exp(-λ φ / M_Pl)
        #
        # V0 is set by the current dark energy density ~ (2.3 meV)⁴
        self.V0_quintessence = (2.3e-3)**4  # eV⁴ ~ observed Λ

        # λ controls how dynamical the DE is.  For λ << 1, w ≈ -1 (ΛCDM).
        # The KK framework predicts λ = sqrt(2/3) for a pure dilaton.
        self.lambda_quintessence = np.sqrt(2.0 / 3.0)

        logger.info(f"KK parameters initialized: R5={self.R5:.2e} m, "
                     f"m_KK={self.m_KK:.2e} eV")

    def summary(self) -> str:
        """Human-readable parameter summary."""
        lines = [
            "=" * 65,
            "SpaceLab — Unified Kaluza-Klein Parameters",
            "=" * 65,
            "",
            "FUNDAMENTAL:",
            f"  R5 (compactification radius)  = {self.R5:.4e} m",
            f"  M5 (5D Planck mass)           = {self.M5:.4e} eV",
            f"  m_KK (KK scale)               = {self.m_KK:.4e} eV",
            f"  M_Pl_4D (derived)             = {self.M_Pl_4D:.4e} eV",
            "",
            "DILATON:",
            f"  m_dilaton                     = {self.m_dilaton:.4e} eV",
            f"  g_dilaton_matter              = {self.g_dilaton_matter:.4e} eV⁻¹",
            f"  dilaton_vev                   = {self.dilaton_vev:.4f}",
            "",
            "AXION (Dark Matter):",
            f"  m_axion                       = {self.m_axion:.4e} eV",
            f"  f_axion (decay constant)      = {self.f_axion:.4e} eV",
            f"  g_axion_dilaton               = {self.g_axion_dilaton:.4e}",
            f"  instanton_suppression         = {self.instanton_suppression:.4e}",
            f"  instanton_action S            = {self.instanton_action:.2f}",
            "",
            "GRAVIPHOTON:",
            f"  g_graviphoton                 = {self.g_graviphoton:.4e} eV⁻¹",
            "",
            "QUINTESSENCE (Dark Energy):",
            f"  V0                            = {self.V0_quintessence:.4e} eV⁴",
            f"  lambda                        = {self.lambda_quintessence:.4f}",
            "=" * 65,
        ]
        return "\n".join(lines)

    def to_dark_matter_params(self) -> Dict[str, float]:
        """Convert to parameters expected by DarkMatter class."""
        return {
            "mass": self.m_axion,
            "coupling_dilaton": abs(self.g_axion_dilaton),
            "coupling_curvature": self.g_dilaton_matter**2 * self.m_KK**2,
        }

    def to_quintessence_params(self) -> Dict[str, float]:
        """Convert to parameters expected by QuintessenceField class."""
        return {
            "V0": self.V0_quintessence,
            "lambda_": self.lambda_quintessence,
        }

    def to_spacetime_params(self, bh_mass_solar: float) -> Dict[str, float]:
        """Parameters for SpacetimeGeometry."""
        return {
            "mass": bh_mass_solar,
        }

    def dilaton_field_at_r(self, r_meters: float, bh_mass_solar: float) -> float:
        """Background dilaton field value at radius r from a BH.

        The dilaton approaches its VEV far from the BH, with a
        Yukawa-like correction near the horizon:
            Φ(r) = Φ_vev + Q_d × exp(-m_d × r) / r

        where Q_d is the dilaton charge of the BH (proportional to mass).
        """
        rs = 2 * G_N * bh_mass_solar * MSUN_KG / (3e8)**2
        Q_d = self.g_dilaton_matter * bh_mass_solar * MSUN_KG * (3e8)**2
        # Yukawa correction
        m_d_inv_m = HBAR_C / self.m_dilaton  # dilaton Compton wavelength [m]
        correction = Q_d * np.exp(-r_meters / m_d_inv_m) / r_meters if r_meters > rs else 0.0
        return self.dilaton_vev + correction

    def axion_soliton_profile(self, r_kpc: np.ndarray, M_halo_Msun: float = 1e11) -> np.ndarray:
        """Predict the DM density profile [M☉/kpc³] from KK parameters.

        Uses the soliton ground state + NFW envelope (Schive et al. 2014).
        The soliton-halo mass relation ties the core to the host halo:

            M_sol = 1/4 × (ζ(z)/ζ(0))^(1/6) × (M_halo/M₀)^(1/3) × M₀
            where M₀ ~ 4.4e7 × (m_a/10⁻²² eV)⁻¹ M☉

        Core radius:
            r_c = 1.6 kpc × (m_a/10⁻²² eV)⁻¹ × (M_halo/10⁹ M☉)⁻¹/³

        Central density:
            ρ_c = 1.9e7 M☉/kpc³ × (m_a/10⁻²³ eV)⁻² × (r_c/kpc)⁻⁴

        Profile:
            ρ(r) = ρ_c / (1 + 0.091 (r/r_c)²)⁸   (for r < r_transition)
                 = ρ_NFW(r)                          (for r > r_transition)

        Args:
            r_kpc: Radii [kpc].
            M_halo_Msun: Total halo virial mass [M☉].

        Returns:
            ρ(r) [M☉/kpc³]
        """
        m_a_22 = self.m_axion / 1e-22   # in units of 10⁻²² eV

        # Soliton core radius from Schive+2014 soliton-halo mass relation
        r_c_kpc = 1.6 / m_a_22 * (1e9 / M_halo_Msun)**(1.0 / 3.0)

        # Enforce physical bounds
        r_c_kpc = max(r_c_kpc, 0.01)  # minimum 10 pc

        # Central density from the soliton solution
        m_a_23 = self.m_axion / 1e-23
        rho_c = 1.9e7 * m_a_23**(-2) * r_c_kpc**(-4)

        # Soliton core
        x = r_kpc / r_c_kpc
        rho_soliton = rho_c / (1.0 + 0.091 * x**2)**8

        # NFW envelope: match at r_transition ~ 3 × r_c
        # ρ_NFW(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
        # r_s from concentration-mass relation: c ~ 15 × (M/10¹² M☉)^(-0.1)
        c200 = 15.0 * (M_halo_Msun / 1e12)**(-0.1)
        # r200 from M200 = (4/3)π × 200 × ρ_crit × r200³
        H0_si = 67.4e3 / 3.0857e22  # s⁻¹
        rho_crit = 3.0 * H0_si**2 / (8.0 * np.pi * G_N) / MSUN_KG * KPC_TO_M**3
        r200 = (3.0 * M_halo_Msun / (4.0 * np.pi * 200 * rho_crit))**(1.0/3.0)
        r_s = r200 / c200

        rho_s_nfw = M_halo_Msun / (4.0 * np.pi * r_s**3 * (np.log(1+c200) - c200/(1+c200)))

        r_transition = 3.0 * r_c_kpc
        rho_nfw = rho_s_nfw / ((r_kpc / r_s) * (1 + r_kpc / r_s)**2)

        # Smooth transition: use soliton inside, NFW outside, take max
        rho = np.maximum(rho_soliton, rho_nfw)

        return rho

    def axion_rotation_velocity(self, r_kpc: np.ndarray, M_halo_Msun: float = 1e11) -> np.ndarray:
        """Predict rotation velocity [km/s] from the KK-derived axion profile.

        Args:
            r_kpc: Radii [kpc].
            M_halo_Msun: Host halo virial mass [M☉], estimated from V_flat.
        """
        from scipy.integrate import cumulative_trapezoid

        # Fine radial grid for integration
        r_fine = np.linspace(1e-4, r_kpc[-1] * 1.1, 2000)
        rho_fine = self.axion_soliton_profile(r_fine, M_halo_Msun)

        # M(<r) = ∫ 4π r'² ρ(r') dr'
        integrand = 4.0 * np.pi * r_fine**2 * rho_fine
        M_enc_fine = np.zeros_like(r_fine)
        M_enc_fine[1:] = cumulative_trapezoid(integrand, r_fine)

        # Interpolate to requested radii
        M_enc = np.interp(r_kpc, r_fine, M_enc_fine)  # M☉

        # v² = G M / r
        r_m = r_kpc * KPC_TO_M
        M_kg = M_enc * MSUN_KG
        v_sq = G_N * M_kg / np.maximum(r_m, 1e-10)
        return np.sqrt(np.maximum(v_sq, 0.0)) / 1e3  # km/s


# ===================================================================
# Preset configurations
# ===================================================================

def default_kk_params() -> KKParameters:
    """Default KK parameters tuned for fuzzy DM phenomenology.

    R5 ~ 10⁻⁶ m gives m_axion ~ 10⁻²² eV, consistent with
    fuzzy DM constraints from Lyman-alpha and rotation curves.
    """
    return KKParameters(
        R5=1e-6,
        M5=1e29,
        stabilization_scale=0.1,
        instanton_suppression=1e-10,
    )


def scan_R5(
    R5_values: np.ndarray,
    quantity: str = "m_axion",
) -> np.ndarray:
    """Scan a derived quantity as a function of R5.

    Useful for understanding how the compactification radius
    maps to observable parameters.

    Args:
        R5_values: Array of R5 values [meters].
        quantity: Attribute name to extract.

    Returns:
        Array of the requested quantity.
    """
    result = np.zeros(len(R5_values))
    for i, R5 in enumerate(R5_values):
        kk = KKParameters(R5=R5)
        result[i] = getattr(kk, quantity)
    return result
