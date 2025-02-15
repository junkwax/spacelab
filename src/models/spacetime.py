import numpy as np
from typing import Union, Tuple

class SpacetimeGeometry:
    """General relativistic spacetime geometry toolkit.

    Args:
        mass (float): Black hole mass [solar masses]
    """
    def __init__(self, mass: float):
        if mass <= 0:
            raise ValueError("Mass must be positive.")
        self.mass = mass  # mass in solar masses
        self.G = 6.67430e-11  # Gravitational constant [m^3 kg^-1 s^-2]
        self.c = 299792458  # Speed of light [m/s]

    def schwarzschild_metric(
        self,
        r: Union[float, np.ndarray]
    ) -> Union[Tuple[float, float, float, float], Tuple[np.ndarray,...]]:
        """Compute Schwarzschild metric components in spherical coordinates.

        Args:
            r: Radial coordinate [meters]

        Returns:
            Tuple containing (g_tt, g_rr, g_theta_theta, g_phi_phi)
        """
        # Convert mass from solar masses to kilograms
        mass_kg = self.mass * 1.98847e30  # 1 solar mass = 1.988e30 kg

        # Calculate Schwarzschild radius in meters
        rs = 2 * self.G * mass_kg / (self.c**2)

        # Ensure radius is greater than Schwarzschild radius
        if np.any(r <= rs):
            raise ValueError("Radius must be greater than the Schwarzschild radius.")

        # Schwarzschild metric components
        g_tt = -(1 - rs / r)
        g_rr = 1 / (1 - rs / r)
        g_theta_theta = r**2
        g_phi_phi = r**2 * np.sin(np.pi/2)**2  # Assume equatorial plane

        return (g_tt, g_rr, g_theta_theta, g_phi_phi)

    def ricci_curvature(
        self,
        r: Union[float, np.ndarray],
        dilaton_field: Union[float, np.ndarray],
        graviphoton_field: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Compute Ricci scalar curvature for the Kaluza-Klein metric.

        Args:
            r: Radial coordinate [meters]
            dilaton_field: Dilaton field value
            graviphoton_field: Graviphoton field value

        Returns:
            Ricci scalar [m^-2]
        """
        # Convert r to a NumPy array if it's a list
        if isinstance(r, list):
            r = np.array(r)

        # Calculate metric components
        g_tt, g_rr, g_theta_theta, g_phi_phi, g_yy, g_ty = self.kaluza_klein_metric(r, dilaton_field, graviphoton_field)

        # Calculate derivatives of metric components (assuming spherical symmetry and time independence)
        d_g_tt_dr = np.gradient(g_tt, r)
        d_g_rr_dr = np.gradient(g_rr, r)
        #... (Calculate other derivatives as needed)

        # Calculate Ricci scalar (simplified for spherical symmetry and time independence)
        R = (
            -1 / (2 * g_rr**2) * (d_g_tt_dr / g_tt * d_g_rr_dr - 2 * d_g_tt_dr**2 / g_tt**2 + 2 * np.gradient(d_g_tt_dr, r) / g_tt)
            #... (Add other terms based on the full Ricci scalar expression)
        )
        return R