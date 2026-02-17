import numpy as np
from typing import Union, Tuple

class SpacetimeGeometry:
    """General relativistic spacetime geometry toolkit.

    Args:
        mass (float): Black hole mass [solar masses]
    """
    # Constants
    G = 6.67430e-11       # Gravitational constant [m^3 kg^-1 s^-2]
    C = 299792458.0       # Speed of light [m/s]
    SOLAR_MASS_KG = 1.98847e30

    def __init__(self, mass: float):
        if mass <= 0:
            raise ValueError("Mass must be positive.")
        self.mass = mass

    def _get_schwarzschild_radius(self) -> float:
        """Calculate the Schwarzschild radius in meters."""
        mass_kg = self.mass * self.SOLAR_MASS_KG
        return 2 * self.G * mass_kg / (self.C ** 2)

    def schwarzschild_metric(
        self,
        r: Union[float, np.ndarray, list],
        theta: float = np.pi / 2,
    ) -> Tuple[Union[float, np.ndarray], ...]:
        """Compute Schwarzschild metric components in spherical coordinates.

        Args:
            r: Radial coordinate [meters]. Must be > Schwarzschild radius.
            theta: Polar angle [radians]. Defaults to π/2 (equatorial plane).

        Returns:
            Tuple of (g_tt, g_rr, g_theta_theta, g_phi_phi).
        """
        r_val = np.asarray(r, dtype=float)
        rs = self._get_schwarzschild_radius()

        if np.any(r_val <= rs):
            raise ValueError(
                f"Radius must be greater than the Schwarzschild radius ({rs:.2e} m)."
            )

        schwarz_factor = 1.0 - rs / r_val
        g_tt = -schwarz_factor
        g_rr = 1.0 / schwarz_factor
        g_theta_theta = r_val ** 2
        g_phi_phi = r_val ** 2 * np.sin(theta) ** 2

        return (g_tt, g_rr, g_theta_theta, g_phi_phi)

    def kaluza_klein_metric(
        self,
        r: Union[float, np.ndarray, list],
        dilaton_field: Union[float, np.ndarray],
        graviphoton_field: Union[float, np.ndarray],
        theta: float = np.pi / 2,
    ) -> Tuple[np.ndarray, ...]:
        """Compute Kaluza-Klein metric components.

        Args:
            r: Radial coordinate [meters]. Must be > Schwarzschild radius.
            dilaton_field: Dilaton scalar field value(s).
            graviphoton_field: Graviphoton field value(s).
            theta: Polar angle [radians]. Defaults to π/2 (equatorial plane).

        Returns:
            Tuple of (g_tt, g_rr, g_theta_theta, g_phi_phi, g_yy, g_ty).
        """
        r_val = np.asarray(r, dtype=float)
        d_val = np.asarray(dilaton_field, dtype=float)
        gp_val = np.asarray(graviphoton_field, dtype=float)

        rs = self._get_schwarzschild_radius()

        if np.any(r_val <= rs):
            raise ValueError("Radius must be greater than the Schwarzschild radius.")

        exp_d = np.exp(-d_val / 2.0)
        schwarz_factor = 1.0 - rs / r_val

        g_tt = -schwarz_factor * exp_d
        g_rr = (1.0 / schwarz_factor) * exp_d
        g_theta_theta = (r_val ** 2) * exp_d
        g_phi_phi = (r_val ** 2 * np.sin(theta) ** 2) * exp_d
        g_yy = np.exp(d_val)
        g_ty = gp_val

        return (g_tt, g_rr, g_theta_theta, g_phi_phi, g_yy, g_ty)

    def ricci_curvature(
        self,
        r: Union[float, np.ndarray, list],
        dilaton_field: Union[float, np.ndarray],
        graviphoton_field: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute Ricci scalar curvature for the Kaluza-Klein metric.

        For scalar inputs, computes the analytic Schwarzschild Kretschner-
        approximated curvature.  For array inputs, uses numerical gradients.
        """
        r_val = np.asarray(r, dtype=float)

        # Scalar case: use analytic approximation R ~ 12 rs^2 / r^6
        if np.ndim(r_val) == 0:
            rs = self._get_schwarzschild_radius()
            if r_val <= rs:
                raise ValueError("Radius must be greater than the Schwarzschild radius.")
            return 12.0 * rs ** 2 / r_val ** 6

        g_tt, g_rr, _, _, _, _ = self.kaluza_klein_metric(
            r_val, dilaton_field, graviphoton_field
        )

        d_g_tt_dr = np.gradient(g_tt, r_val)
        d_g_rr_dr = np.gradient(g_rr, r_val)

        with np.errstate(divide="ignore", invalid="ignore"):
            term1 = d_g_tt_dr / g_tt
            term2 = d_g_rr_dr
            R = -1.0 / (2.0 * g_rr ** 2) * (term1 * term2)

        return np.nan_to_num(R)
