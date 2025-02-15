import numpy as np
from typing import Union, overload

class DarkMatter:
    """Model for axion-like dark matter in higher-dimensional spacetime.
    
    Args:
        mass (float): Mass of the dark matter particle in eV.
        coupling (float): Coupling constant to the dilaton field.
    """
    def __init__(self, mass: float, coupling: float):
        if mass <= 0 or coupling <= 0:
            raise ValueError("Mass and coupling must be positive.")
        self.mass = mass
        self.coupling = coupling

    @overload
    def density_profile(self, r: float) -> float: ...

    @overload
    def density_profile(self, r: np.ndarray) -> np.ndarray: ...

    def density_profile(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the dark matter density profile.
        
        Args:
            r (float or array-like): Radial distance from the black hole.
        
        Returns:
            np.ndarray: Density at radius `r`.
        
        Raises:
            ValueError: If `r` is non-positive.
        """
        r_arr = np.asarray(r)
        if np.any(r_arr <= 0):
            raise ValueError("Radius `r` must be positive.")
        return self.coupling * np.exp(-self.mass * r_arr) / r_arr**2