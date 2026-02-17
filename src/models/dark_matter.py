import numpy as np
from typing import Union, Tuple

# Minimum radius to avoid numerical blow-up in 1/r^2 terms.
MIN_RADIUS = 1e-10


class DarkMatter:
    """Model for axion-like dark matter.

    Args:
        mass: Axion mass parameter [eV].
        coupling_dilaton: Coupling to the dilaton field.
        coupling_curvature: Coupling to spacetime curvature (Ricci scalar).
    """

    def __init__(self, mass: float, coupling_dilaton: float, coupling_curvature: float):
        if mass <= 0 or coupling_dilaton <= 0 or coupling_curvature <= 0:
            raise ValueError("Mass and coupling constants must be positive.")
        self.mass = mass
        self.coupling_dilaton = coupling_dilaton
        self.coupling_curvature = coupling_curvature

    def density_profile(
        self,
        r: Union[float, np.ndarray],
        dilaton_field: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute the dark matter density profile.

        Args:
            r: Radial distance.  Must be > 0 (clamped to MIN_RADIUS internally).
            dilaton_field: Value of the dilaton field at the given radius.

        Returns:
            Dark matter density at the specified radius.
        """
        r_arr = np.asarray(r, dtype=float)
        if np.any(r_arr <= 0):
            raise ValueError("Radius `r` must be positive.")

        # Clamp to avoid catastrophic 1/r^2 blow-up for very small r.
        r_safe = np.maximum(r_arr, MIN_RADIUS)
        return self.coupling_dilaton * np.exp(-self.mass * r_safe) * dilaton_field / r_safe ** 2

    def potential(
        self,
        phi: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        dilaton_field: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Effective scalar field potential.

        Includes mass term, non-minimal curvature coupling, and dilaton coupling.
        """
        # Placeholder Ricci scalar (flat-space approximation)
        R = 2.0 / np.asarray(r, dtype=float) ** 2
        return (
            0.5 * self.mass ** 2 * phi ** 2
            + self.coupling_curvature * R * phi ** 2
            + self.coupling_dilaton * dilaton_field * phi ** 2
        )

    def field_equation(
        self,
        y: Tuple[Union[float, np.ndarray], Union[float, np.ndarray]],
        r: Union[float, np.ndarray],
        dilaton_field: Union[float, np.ndarray],
        graviphoton_field: Union[float, np.ndarray],
    ) -> Tuple[float, float]:
        """Right-hand side of the scalar field ODE in the form expected by ODE solvers.

        The scalar field equation is rewritten as a first-order system:
            dphi/dr   = dphi_dr          (pass-through)
            d²phi/dr² = -2/r dphi_dr - V(phi, r)

        Args:
            y: Tuple of (phi, dphi_dr) — current field value and its radial derivative.
            r: Radial coordinate.
            dilaton_field: Dilaton field value at r.
            graviphoton_field: Graviphoton field value at r (reserved for future use).

        Returns:
            Tuple of (dphi_dr, d²phi/dr²) as plain floats.
        """
        phi, dphi_dr = y
        ddphi_dr2 = -2.0 / r * dphi_dr - self.potential(phi, r, dilaton_field)

        # Return plain floats so callers don't need to unwrap arrays.
        return (float(dphi_dr), float(ddphi_dr2))
