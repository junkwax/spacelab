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
        self.solar_mass_kg = 1.98847e30 # kg

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
        mass_kg = self.mass * self.solar_mass_kg  # 1 solar mass = 1.988e30 kg

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

    def kaluza_klein_metric(
        self,
        r: Union[float, np.ndarray],
        dilaton_field: Union[float, np.ndarray],
        graviphoton_field: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray,...]:
        """Compute Kaluza-Klein metric components.

        Args:
            r: Radial coordinate [meters]
            dilaton_field: Dilaton field value
            graviphoton_field: Graviphoton field value

        Returns:
            Tuple containing (g_tt, g_rr, g_theta_theta, g_phi_phi, g_yy, g_ty)
        """
        # Convert mass from solar masses to kilograms
        mass_kg = self.mass * self.solar_mass_kg

        # Calculate Schwarzschild radius in meters
        rs = 2 * self.G * mass_kg / (self.c**2)

        # Ensure radius is greater than Schwarzschild radius
        if np.any(r <= rs):
            raise ValueError("Radius must be greater than the Schwarzschild radius.")

        # Kaluza-Klein metric components
        g_tt = -(1 - rs / r) * np.exp(-dilaton_field / 2)
        g_rr = 1 / (1 - rs / r) * np.exp(-dilaton_field / 2)
        g_theta_theta = r**2 * np.exp(-dilaton_field / 2)
        g_phi_phi = r**2 * np.sin(np.pi/2)**2 * np.exp(-dilaton_field / 2)  # Assume equatorial plane
        g_yy = np.exp(dilaton_field)
        g_ty = graviphoton_field

        return (g_tt, g_rr, g_theta_theta, g_phi_phi, g_yy, g_ty)

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
        r = np.asarray(r)
        dilaton_field = np.asarray(dilaton_field)
        graviphoton_field = np.asarray(graviphoton_field)

        # Convert mass from solar masses to kilograms and calculate rs
        mass_kg = self.mass * self.solar_mass_kg
        rs = 2 * self.G * mass_kg / (self.c**2)

        # Ensure radius is greater than Schwarzschild radius
        if np.any(r <= rs):
            raise ValueError("Radius must be greater than the Schwarzschild radius.")

        # Calculate derivatives using central differences
        d_dilaton_dr = np.gradient(dilaton_field, r)
        d2_dilaton_dr2 = np.gradient(d_dilaton_dr, r)
        d_graviphoton_dr = np.gradient(graviphoton_field, r)
        d2_graviphoton_dr2 = np.gradient(d_graviphoton_dr, r)
        
        # Ricci scalar expression (derived from SymPy)
        R = (-graviphoton_field**2 * np.exp(dilaton_field/2)
             - 4.0 * graviphoton_field * np.exp(dilaton_field/2) * d_graviphoton_dr
             - 4.0 * np.exp(dilaton_field/2) * d2_graviphoton_dr2
             + 1.0 * r * np.exp(dilaton_field/2) * d_graviphoton_dr**2
             - 1.0 * (r - rs) * np.exp(dilaton_field/2) * d_dilaton_dr**2
             - 4.0 * (r - rs) * np.exp(dilaton_field/2) * d2_dilaton_dr2
             - 8.0 * np.exp(dilaton_field/2) * d_dilaton_dr) / (4.0 * r * (r - rs))
        return R
