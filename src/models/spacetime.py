import numpy as np
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)

class SpacetimeGeometry:
    """General relativistic spacetime geometry toolkit.
    
    Args:
        mass (float): Black hole mass [solar masses]
    """
    def __init__(self, mass: float):
        if mass <= 0:
            logger.error("Invalid black hole mass: %s", mass)
            raise ValueError("Mass must be positive.")
        self.mass = mass
        self.G = 6.67430e-11  # Gravitational constant [m^3 kg^-1 s^-2]
        self.c = 299792458  # Speed of light [m/s]

    def schwarzschild_metric(
        self,
        r: Union[float, np.ndarray]
    ) -> Union[Tuple[float, float, float, float], Tuple[np.ndarray, ...]]:
        """Compute Schwarzschild metric components in spherical coordinates.
        
        Args:
            r: Radial coordinate [meters]
        
        Returns:
            Tuple containing (g_tt, g_rr, g_θθ, g_φφ)
        """
        rs = 2 * self.G * self.mass * 1.988e30 / (self.c**2)  # Schwarzschild radius
        g_tt = -(1 - rs / r)
        g_rr = 1 / g_tt
        g_theta_theta = r**2
        g_phi_phi = r**2 * np.sin(np.pi/2)**2  # Assume equatorial plane
        
        return (g_tt, g_rr, g_theta_theta, g_phi_phi)

    def ricci_curvature(
        self,
        r: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Compute Ricci scalar curvature for Schwarzschild spacetime.
        
        Args:
            r: Radial coordinate [meters]
        
        Returns:
            Ricci scalar [m^-2]
        """
        # Schwarzschild spacetime is Ricci-flat (R=0)
        return np.zeros_like(r)