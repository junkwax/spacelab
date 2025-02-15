import numpy as np

class DarkMatter:
    def __init__(self, mass, coupling):
        self.mass = mass  # Dark matter mass
        self.coupling = coupling  # Coupling constant

    def density_profile(self, r):
        """Compute dark matter density profile."""
        return self.coupling * np.exp(-self.mass * r) / r**2

    def pressure(self, r, curvature):
        """Compute dark matter pressure."""
        return -2 * self.coupling * curvature * np.exp(-self.mass * r)