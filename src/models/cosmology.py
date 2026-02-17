"""
FRW cosmological background with coupled quintessence dark energy.

Solves the system:
    H² = (8πG / 3) (ρ_m + ρ_r + ρ_DE)
    φ̈  = -3H φ̇ - dV/dφ
    ȧ   = H a

where ρ_DE = ½φ̇² + V(φ), and matter/radiation dilute as a⁻³ / a⁻⁴.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

G_N = 6.67430e-11           # m³ kg⁻¹ s⁻²
C = 299792458.0              # m/s
H0_DEFAULT = 67.4            # km/s/Mpc  (Planck 2018)
MPC_TO_M = 3.0857e22         # meters per Mpc


class FRWCosmology:
    """Friedmann-Robertson-Walker background with quintessence.

    Evolves the scale factor a(t) and quintessence field φ(t)
    self-consistently from some initial redshift to today (a=1).

    Args:
        H0: Hubble constant [km/s/Mpc].
        Omega_m0: Present-day matter density fraction.
        Omega_r0: Present-day radiation density fraction.
        quintessence: A QuintessenceField instance (or None for ΛCDM).
    """

    def __init__(
        self,
        H0: float = H0_DEFAULT,
        Omega_m0: float = 0.315,
        Omega_r0: float = 9.1e-5,
        quintessence=None,
    ):
        self.H0 = H0
        self.H0_si = H0 * 1e3 / MPC_TO_M
        self.Omega_m0 = Omega_m0
        self.Omega_r0 = Omega_r0
        self.quintessence = quintessence

        # Critical density today  ρ_c = 3H₀²/(8πG)
        self.rho_crit0 = 3.0 * self.H0_si**2 / (8.0 * np.pi * G_N)
        self.rho_m0 = Omega_m0 * self.rho_crit0
        self.rho_r0 = Omega_r0 * self.rho_crit0

    def _matter_density(self, a: float) -> float:
        return self.rho_m0 / a**3

    def _radiation_density(self, a: float) -> float:
        return self.rho_r0 / a**4

    def _de_density(self, phi: float, dphi_dt: float) -> float:
        if self.quintessence is None:
            Omega_de0 = 1.0 - self.Omega_m0 - self.Omega_r0
            return Omega_de0 * self.rho_crit0
        V = self.quintessence.potential(phi)
        return 0.5 * dphi_dt**2 + V

    def hubble(self, a: float, phi: float = 0.0, dphi_dt: float = 0.0) -> float:
        """H(a) = sqrt(8πG/3 × ρ_total) in s⁻¹."""
        rho_total = (
            self._matter_density(a)
            + self._radiation_density(a)
            + self._de_density(phi, dphi_dt)
        )
        return np.sqrt(8.0 * np.pi * G_N / 3.0 * np.abs(rho_total))

    def _ode_rhs(self, t, y):
        """RHS for the coupled {a, φ, φ̇} system in cosmic time."""
        a, phi, dphi_dt = y
        H = self.hubble(a, phi, dphi_dt)

        da_dt = H * a

        if self.quintessence is not None:
            ddphi_dt2 = self.quintessence.field_acceleration(phi, dphi_dt, H)
        else:
            ddphi_dt2 = 0.0

        return [da_dt, dphi_dt, ddphi_dt2]

    def evolve(
        self,
        a_start: float = 0.001,
        a_end: float = 1.0,
        phi_init: float = 0.0,
        dphi_dt_init: float = 0.0,
        n_points: int = 1000,
        method: str = "RK45",
    ) -> Dict[str, np.ndarray]:
        """Evolve the cosmological background from a_start to a_end.

        Returns:
            Dict with 't', 'a', 'z', 'phi', 'dphi_dt', 'H', 'w_de', 'Omega_de'.
        """
        y0 = [a_start, phi_init, dphi_dt_init]
        t_hubble = 1.0 / self.H0_si
        t_span = (0.0, 3.0 * t_hubble)

        def event_a_end(t, y):
            return y[0] - a_end
        event_a_end.terminal = True

        logger.info(
            f"Evolving FRW: a=[{a_start}, {a_end}], "
            f"phi_init={phi_init}, method={method}"
        )

        sol = solve_ivp(
            self._ode_rhs,
            t_span,
            y0,
            method=method,
            events=event_a_end,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success and len(sol.t) < 2:
            raise RuntimeError(f"FRW integration failed: {sol.message}")

        t_final = sol.t[-1]
        t_eval = np.linspace(0, t_final, n_points)
        y_eval = sol.sol(t_eval)

        a_arr = y_eval[0]
        phi_arr = y_eval[1]
        dphi_arr = y_eval[2]
        z_arr = 1.0 / a_arr - 1.0

        H_arr = np.array([
            self.hubble(a, p, dp) for a, p, dp in zip(a_arr, phi_arr, dphi_arr)
        ])

        w_arr = np.zeros_like(a_arr)
        Omega_de_arr = np.zeros_like(a_arr)
        for i in range(len(a_arr)):
            rho_de = self._de_density(phi_arr[i], dphi_arr[i])
            rho_tot = (
                self._matter_density(a_arr[i])
                + self._radiation_density(a_arr[i])
                + rho_de
            )
            Omega_de_arr[i] = rho_de / rho_tot if rho_tot > 0 else 0.0
            if self.quintessence is not None:
                w_arr[i] = self.quintessence.equation_of_state(phi_arr[i], dphi_arr[i])
            else:
                w_arr[i] = -1.0

        logger.info(
            f"Evolution complete. Final z={z_arr[-1]:.4f}, "
            f"w_DE(z=0)={w_arr[-1]:.4f}, Ω_DE(z=0)={Omega_de_arr[-1]:.4f}"
        )

        return {
            "t": t_eval,
            "a": a_arr,
            "z": z_arr,
            "phi": phi_arr,
            "dphi_dt": dphi_arr,
            "H": H_arr,
            "w_de": w_arr,
            "Omega_de": Omega_de_arr,
        }
