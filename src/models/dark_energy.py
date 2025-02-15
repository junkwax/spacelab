class DarkEnergy:
    def __init__(self, V0, lambda_):
        self.V0 = V0  # Energy scale
        self.lambda_ = lambda_  # Slope parameter

    def potential(self, phi):
        """Compute quintessence potential."""
        return self.V0 * np.exp(-self.lambda_ * phi)

    def equation_of_state(self, phi, dphi_dt):
        """Compute equation of state parameter."""
        kinetic = 0.5 * dphi_dt**2
        potential = self.potential(phi)
        return (kinetic - potential) / (kinetic + potential)