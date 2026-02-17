"""
Bulk stress-energy tensor from 5D Kaluza-Klein reduction.

When reducing from 5D to 4D, the Einstein equations acquire extra
source terms from the higher-dimensional geometry:

    G_μν^(4D) = 8πG₄ T_μν^(eff)

where T_μν^(eff) = T_μν^(matter) + T_μν^(dilaton) + T_μν^(axion)
                  + T_μν^(graviphoton) + T_μν^(bulk)

The bulk terms arise from:
    1. Dilaton kinetic + potential energy
    2. Graviphoton field strength F_μν
    3. Axion field on the KK background
    4. Extrinsic curvature of the compactified dimension

This module computes all components given the field values.
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class StressEnergyKK:
    """Effective 4D stress-energy tensor from 5D KK reduction.

    All computations are in SI units unless noted.

    Args:
        kk_params: A KKParameters instance providing the couplings.
    """

    def __init__(self, kk_params):
        self.kk = kk_params

    def dilaton_stress_energy(
        self,
        phi: float,
        dphi_dr: float,
        dphi_dt: float = 0.0,
    ) -> Dict[str, float]:
        """Stress-energy from the dilaton field.

        T_μν^(dilaton) = ∂_μ φ ∂_ν φ - g_μν [½(∂φ)² + V(φ)]

        For a static, spherically symmetric configuration:
            ρ_d = ½ (dφ/dr)² + V(φ)
            p_d = ½ (dφ/dr)² - V(φ)

        The dilaton potential is a stabilization potential:
            V(φ) = ½ m_d² (φ - φ_vev)²
        """
        m_d = self.kk.m_dilaton
        phi_vev = self.kk.dilaton_vev

        V = 0.5 * m_d**2 * (phi - phi_vev)**2
        kinetic = 0.5 * (dphi_dr**2 + dphi_dt**2)

        rho = kinetic + V
        p_r = kinetic - V       # radial pressure
        p_t = -kinetic - V      # tangential pressure (anisotropic)

        return {
            "rho": rho,
            "p_radial": p_r,
            "p_tangential": p_t,
            "V": V,
            "kinetic": kinetic,
            "w": p_r / rho if rho > 0 else -1.0,
        }

    def axion_stress_energy(
        self,
        psi: float,
        dpsi_dr: float,
        dilaton_field: float,
    ) -> Dict[str, float]:
        """Stress-energy from the axion dark matter field.

        The axion Lagrangian on the KK background:
            L = ½ exp(-g_d φ) [(∂ψ)² - m_a² ψ²] - g_ad φ ψ²

        This gives:
            ρ_a = ½ (dψ/dr)² + ½ m_a² ψ² + g_ad φ ψ²
            p_a = ½ (dψ/dr)² - ½ m_a² ψ² - g_ad φ ψ²
        """
        m_a = self.kk.m_axion
        g_ad = self.kk.g_axion_dilaton

        kinetic = 0.5 * dpsi_dr**2
        mass_term = 0.5 * m_a**2 * psi**2
        coupling_term = g_ad * dilaton_field * psi**2

        rho = kinetic + mass_term + coupling_term
        p = kinetic - mass_term - coupling_term

        return {
            "rho": rho,
            "p": p,
            "kinetic": kinetic,
            "mass_term": mass_term,
            "coupling_term": coupling_term,
            "w": p / rho if rho > 0 else 0.0,
        }

    def graviphoton_stress_energy(
        self,
        A_t: float,
        dA_dr: float,
    ) -> Dict[str, float]:
        """Stress-energy from the graviphoton field A_μ.

        The graviphoton has a Maxwell-like action:
            L = -¼ exp(√3 φ) F_μν F^μν

        For a static, radial electric-type field (F_tr = dA_t/dr):
            ρ_gp = ½ (dA_t/dr)²
            p_r  = ½ (dA_t/dr)²
            p_t  = -½ (dA_t/dr)²
        """
        E_sq = dA_dr**2

        return {
            "rho": 0.5 * E_sq,
            "p_radial": 0.5 * E_sq,
            "p_tangential": -0.5 * E_sq,
        }

    def bulk_correction(
        self,
        phi: float,
        dphi_dr: float,
        r: float,
    ) -> Dict[str, float]:
        """Bulk stress-energy correction from the extra dimension.

        The extrinsic curvature of the compactified S¹ contributes:
            T_μν^(bulk) = -1/(8πG) × [K_μα K^α_ν - K K_μν
                          - ½ g_μν (K_αβ K^αβ - K²)]

        For an S¹ with radius R(r) = R5 × exp(φ/√3):
            K_ij = -(1/2R) (dR/dr) δ_ij   (extrinsic curvature)

        Leading to an effective correction:
            ρ_bulk = (3/2) (dφ/dr)² / (8πG R5²) × exp(-2φ/√3)
        """
        from src.models.kaluza_klein import G_N

        exp_factor = np.exp(-2.0 * phi / np.sqrt(3.0))
        R5 = self.kk.R5

        rho_bulk = (3.0 / 2.0) * dphi_dr**2 / (8.0 * np.pi * G_N * R5**2) * exp_factor

        # The bulk pressure is anisotropic
        p_bulk = -rho_bulk / 3.0  # equation of state for bulk curvature

        return {
            "rho": rho_bulk,
            "p": p_bulk,
            "w": -1.0 / 3.0,
        }

    def total_stress_energy(
        self,
        r: float,
        phi: float,
        dphi_dr: float,
        psi: float,
        dpsi_dr: float,
        A_t: float = 0.0,
        dA_dr: float = 0.0,
        dphi_dt: float = 0.0,
    ) -> Dict[str, float]:
        """Compute the total effective stress-energy at radius r.

        Returns:
            Dict with total ρ, p, and individual component breakdowns.
        """
        dilaton = self.dilaton_stress_energy(phi, dphi_dr, dphi_dt)
        axion = self.axion_stress_energy(psi, dpsi_dr, phi)
        gp = self.graviphoton_stress_energy(A_t, dA_dr)
        bulk = self.bulk_correction(phi, dphi_dr, r)

        rho_total = dilaton["rho"] + axion["rho"] + gp["rho"] + bulk["rho"]
        p_total = (
            dilaton["p_radial"] + axion["p"] + gp["p_radial"] + bulk["p"]
        )

        return {
            "rho_total": rho_total,
            "p_total": p_total,
            "w_eff": p_total / rho_total if rho_total > 0 else 0.0,
            "dilaton": dilaton,
            "axion": axion,
            "graviphoton": gp,
            "bulk": bulk,
            "rho_breakdown": {
                "dilaton": dilaton["rho"],
                "axion": axion["rho"],
                "graviphoton": gp["rho"],
                "bulk": bulk["rho"],
            },
        }

    def profile(
        self,
        r_array: np.ndarray,
        phi_array: np.ndarray,
        dphi_dr_array: np.ndarray,
        psi_array: np.ndarray,
        dpsi_dr_array: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute stress-energy profiles along a radial grid.

        Returns:
            Dict of arrays for each component.
        """
        n = len(r_array)
        rho_total = np.zeros(n)
        rho_dilaton = np.zeros(n)
        rho_axion = np.zeros(n)
        rho_bulk = np.zeros(n)
        w_eff = np.zeros(n)

        for i in range(n):
            T = self.total_stress_energy(
                r_array[i], phi_array[i], dphi_dr_array[i],
                psi_array[i], dpsi_dr_array[i],
            )
            rho_total[i] = T["rho_total"]
            rho_dilaton[i] = T["rho_breakdown"]["dilaton"]
            rho_axion[i] = T["rho_breakdown"]["axion"]
            rho_bulk[i] = T["rho_breakdown"]["bulk"]
            w_eff[i] = T["w_eff"]

        return {
            "r": r_array,
            "rho_total": rho_total,
            "rho_dilaton": rho_dilaton,
            "rho_axion": rho_axion,
            "rho_bulk": rho_bulk,
            "w_eff": w_eff,
        }
