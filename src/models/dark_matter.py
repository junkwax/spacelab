import numpy as np

class DarkMatter:
    """Model for axion-like dark matter in higher-dimensional spacetime.
    
    Args:
        mass (float): Mass of the dark matter particle in eV.
        coupling (float): Coupling constant to the dilaton field.
    """
    def __init__(self, mass, coupling):
        if mass <= 0:
            raise ValueError("Mass must be positive.")
        if coupling <= 0:
            raise ValueError("Coupling constant must be positive.")
        self.mass = mass
        self.coupling = coupling

    def density_profile(self, r):
        """Compute the dark matter density profile.
        
        Args:
            r (float or array-like): Radial distance from the black hole.
        
        Returns:
            np.ndarray: Density at radius `r`.
        
        Raises:
            ValueError: If `r` is non-positive.
        """
        r = np.asarray(r)
        if np.any(r <= 0):
            raise ValueError("Radius `r` must be positive.")
        return self.coupling * np.exp(-self.mass * r) / r**2