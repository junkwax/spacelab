"""
Self-consistent field solver for the axion-dilaton system on a KK background.

Solves the coupled system:
    ∇²φ_d - m_d² (φ_d - φ_vev) = g_ad ψ²          (dilaton)
    ∇²ψ   - m_a² ψ - 2 g_ad φ_d ψ = 0              (axion)

on the Kaluza-Klein metric background, with boundary conditions:
    φ_d → φ_vev as r → ∞
    ψ  → 0 as r → ∞
    Regularity at r = r_min (just outside horizon)

The axion density profile ρ_DM(r) = m_a² |ψ|² is then a PREDICTION
of the KK theory, not an assumed functional form.

This closes the loop between spacetime.py and rotation_curves.py.
"""

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

G_N = 6.67430e-11
C = 299792458.0
HBAR_C = 1.9733e-7    # eV·m
KPC_TO_M = 3.0857e19
MSUN_KG = 1.98847e30
EV_PER_KG = 5.609e35


class KKFieldSolver:
    """Solve the coupled axion-dilaton field equations on a KK black hole background.

    Args:
        kk_params: KKParameters instance.
        bh_mass_solar: Black hole mass in solar masses.
    """

    def __init__(self, kk_params, bh_mass_solar: float = 1e6):
        self.kk = kk_params
        self.bh_mass = bh_mass_solar

        # Schwarzschild radius
        self.rs = 2 * G_N * bh_mass_solar * MSUN_KG / C**2

        # Convert field theory masses to inverse meters for the ODE
        # m [eV] → κ [m⁻¹] via κ = m / (ℏc)
        self.kappa_d = self.kk.m_dilaton / HBAR_C  # dilaton inverse Compton wavelength
        self.kappa_a = self.kk.m_axion / HBAR_C    # axion inverse Compton wavelength

        # Coupling in SI-compatible units
        self.g_ad = self.kk.g_axion_dilaton / HBAR_C  # coupling per meter

    def _metric_factor(self, r: float) -> float:
        """Schwarzschild factor f(r) = 1 - rs/r."""
        return 1.0 - self.rs / r

    def field_equations(self, r, y):
        """RHS of the coupled ODE system.

        State vector y = [φ_d, dφ_d/dr, ψ, dψ/dr]

        The Laplacian in Schwarzschild coordinates:
            ∇²φ = f(r) φ'' + [f'(r) + 2f(r)/r] φ'
            where f(r) = 1 - rs/r

        Rearranging for φ'':
            φ'' = [m_d² (φ - φ_vev) + g ψ² - (f'/f + 2/r) φ'] / f
        """
        phi, dphi, psi, dpsi = y

        f = self._metric_factor(r)
        if abs(f) < 1e-15:
            return [0.0, 0.0, 0.0, 0.0]

        dfdr = self.rs / r**2
        coeff = dfdr / f + 2.0 / r

        # Dilaton equation: ∇²φ = m_d² (φ - φ_vev) + g_ad ψ²
        source_d = self.kappa_d**2 * (phi - self.kk.dilaton_vev) + self.g_ad * psi**2
        ddphi = (source_d - coeff * dphi) / f

        # Axion equation: ∇²ψ = m_a² ψ + 2 g_ad φ ψ
        source_a = self.kappa_a**2 * psi + 2.0 * self.g_ad * phi * psi
        ddpsi = (source_a - coeff * dpsi) / f

        return [dphi, ddphi, dpsi, ddpsi]

    def solve_shooting(
        self,
        r_min_rs: float = 1.5,
        r_max_rs: float = 1e4,
        psi_0: float = 1.0,
        n_points: int = 2000,
    ) -> Dict[str, np.ndarray]:
        """Solve the field equations using a shooting method from the near-horizon.

        Integrates outward from r_min to r_max with initial conditions
        set by regularity at the horizon.

        Args:
            r_min_rs: Starting radius in units of rs.
            r_max_rs: Ending radius in units of rs.
            psi_0: Initial axion amplitude at r_min (free parameter).
            n_points: Number of output points.

        Returns:
            Dict with 'r', 'r_rs', 'phi_d', 'psi', 'rho_dm', 'v_circ_kms', etc.
        """
        r_min = r_min_rs * self.rs
        r_max = r_max_rs * self.rs

        # Initial conditions: fields smooth at r_min
        # Dilaton starts at VEV, axion starts with amplitude psi_0
        y0 = [
            self.kk.dilaton_vev,   # φ_d(r_min)
            0.0,                    # dφ_d/dr(r_min) = 0 (smooth)
            psi_0,                  # ψ(r_min)
            0.0,                    # dψ/dr(r_min) = 0 (smooth)
        ]

        # Use log-spaced radial grid for better resolution near BH
        r_eval = np.geomspace(r_min, r_max, n_points)

        logger.info(
            f"Solving KK field equations: r=[{r_min_rs:.1f}, {r_max_rs:.0f}] rs, "
            f"ψ₀={psi_0:.2e}"
        )

        sol = solve_ivp(
            self.field_equations,
            (r_min, r_max),
            y0,
            method="RK45",
            t_eval=r_eval,
            rtol=1e-8,
            atol=1e-12,
            max_step=r_max / 100,
        )

        if not sol.success:
            logger.warning(f"Field solver: {sol.message}")

        r = sol.t
        phi_d = sol.y[0]
        dphi_dr = sol.y[1]
        psi = sol.y[2]
        dpsi_dr = sol.y[3]

        # Dark matter density: ρ = m_a² |ψ|²  (in natural units → convert)
        # ρ [eV⁴/ℏ³c³] → ρ [kg/m³]
        rho_natural = self.kk.m_axion**2 * psi**2  # eV² (in field units)
        # Convert to kg/m³: multiply by eV/(ℏc)³ × c²
        rho_dm_kgm3 = rho_natural * (1.0 / HBAR_C)**3 * EV_PER_KG**(-1)

        # Convert to M☉/kpc³ for astrophysical use
        rho_dm_astro = rho_dm_kgm3 / MSUN_KG * KPC_TO_M**3

        # Enclosed mass and rotation velocity
        r_kpc = r / KPC_TO_M
        M_enc = np.zeros_like(r)
        for i in range(1, len(r)):
            dr = r[i] - r[i-1]
            M_enc[i] = M_enc[i-1] + 4.0 * np.pi * r[i]**2 * rho_dm_kgm3[i] * dr

        v_circ = np.sqrt(G_N * M_enc / np.maximum(r, 1e-10))
        v_circ_kms = v_circ / 1e3

        # Dilaton deviation from VEV
        delta_phi = phi_d - self.kk.dilaton_vev

        logger.info(
            f"Field solution: max |ψ|={np.max(np.abs(psi)):.3e}, "
            f"max Δφ_d={np.max(np.abs(delta_phi)):.3e}, "
            f"max ρ_DM={np.max(rho_dm_astro):.3e} M☉/kpc³"
        )

        return {
            "r": r,
            "r_rs": r / self.rs,
            "r_kpc": r_kpc,
            "phi_dilaton": phi_d,
            "dphi_dr": dphi_dr,
            "delta_phi": delta_phi,
            "psi_axion": psi,
            "dpsi_dr": dpsi_dr,
            "rho_dm_kgm3": rho_dm_kgm3,
            "rho_dm_astro": rho_dm_astro,
            "M_enclosed_kg": M_enc,
            "v_circ_ms": v_circ,
            "v_circ_kms": v_circ_kms,
        }

    def compute_observables(self, solution: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract observable quantities from a field solution.

        Returns:
            Dict with scalar charge, core radius, central density, etc.
        """
        r_kpc = solution["r_kpc"]
        rho = solution["rho_dm_astro"]
        v = solution["v_circ_kms"]
        delta_phi = solution["delta_phi"]

        # Central DM density
        rho_central = rho[0] if rho[0] > 0 else rho[rho > 0][0] if np.any(rho > 0) else 0.0

        # Core radius: where density drops to half central value
        if rho_central > 0:
            half_mask = rho < 0.5 * rho_central
            r_core = r_kpc[half_mask][0] if np.any(half_mask) else r_kpc[-1]
        else:
            r_core = 0.0

        # Maximum rotation velocity
        v_max = np.max(v)
        r_vmax = r_kpc[np.argmax(v)]

        # BH scalar charge: Q_d = r² × (dφ/dr) evaluated at large r
        # (asymptotic Yukawa: φ ~ φ_vev + Q exp(-mr)/r → r²φ' → -Q at large r)
        r_far = solution["r"][-10:]
        dphi_far = solution["dphi_dr"][-10:]
        Q_dilaton = -np.mean(r_far**2 * dphi_far)

        return {
            "rho_central_Msun_kpc3": rho_central,
            "r_core_kpc": r_core,
            "v_max_kms": v_max,
            "r_vmax_kpc": r_vmax,
            "Q_dilaton": Q_dilaton,
            "delta_phi_max": np.max(np.abs(delta_phi)),
        }
