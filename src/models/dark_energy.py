import numpy as np
from typing import Union, overload
import logging

logger = logging.getLogger(__name__)

class QuintessenceField:
    """Model for dynamical dark energy (quintessence) with a scalar field."""
    def __init__(self, V0: float, lambda_: float):
        if V0 <= 0 or lambda_ <= 0:
            logger.error("Invalid quintessence parameters: V0=%s, lambda=%s", V0, lambda_)
            raise ValueError("V0 and lambda must be positive.")
        self.V0 = V0
        self.lambda_ = lambda_
        self.solar_mass_kg = 1.98847e30  # Added
        self.G = 6.67430e-11 # Added
        self.c = 299792458 # Added


    def potential(self, phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the quintessence potential V(φ)."""
        return self.V0 * np.exp(-self.lambda_ * phi)

    def equation_of_state(
        self,
        phi: Union[float, np.ndarray],
        dphi_dt: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Compute the equation of state parameter w = P/ρ."""
        kinetic = 0.5 * dphi_dt**2
        potential = self.potential(phi)
        denominator = kinetic + potential

        if np.any(denominator == 0):
            logger.warning("Division by zero in equation_of_state calculation")
            return np.full_like(denominator, -1.0)  # Return w=-1 for stability

        return (kinetic - potential) / denominator

    def field_equation(self, y, r, phi_dilaton, phi_DM, beta):
        """Quintessence field equation."""
        phi_DE, dphi_DE_dr, dphi_DE_dt = y

        r_arr = np.asarray(r)  # Use r_arr locally
        phi_dilaton = np.asarray(phi_dilaton)
        phi_DM = np.asarray(phi_DM)
        dphi_DE_dr = np.asarray(dphi_DE_dr) # Ensure this is an array!
        dphi_DE_dt = np.asarray(dphi_DE_dt)


        # Calculate derivatives
        d_dilaton_dr = np.gradient(phi_dilaton, r_arr)

        # Convert mass from solar masses to kilograms and calculate rs
        mass_kg = self.mass * self.solar_mass_kg
        rs = 2 * self.G * mass_kg / (self.c**2)

        # Ensure radius is greater than Schwarzschild radius
        if np.any(r_arr <= rs):
            raise ValueError("Radius must be greater than the Schwarzschild radius.")

        # Calculate ddphi_dt2 and ddphi_dr2
        ddphi_dt2 = (1.0 * self.V0 * self.lambda_ * r_arr**2 * np.exp(phi_dilaton / 4) * np.exp(-self.lambda_ * phi_DE)
                    + 1.0 * beta * r_arr**2 * phi_DM**2 * np.exp(phi_dilaton / 4)
                    + 1.0 * dphi_DE_dt * d_dilaton_dr) / (r_arr * (r_arr - rs))

        ddphi_dr2 = (-2.0 * self.V0 * self.lambda_ * r_arr**2 * np.exp(phi_dilaton / 4) * np.exp(-self.lambda_ * phi_DE)
                    - 2.0 * beta * r_arr**2 * phi_DM**2 * np.exp(phi_dilaton / 4)
                    + 8.0 * self.V0 * self.lambda_ * r_arr * np.exp(phi_dilaton / 4) * np.exp(-self.lambda_ * phi_DE)
                    + 8.0 * beta * r_arr * phi_DM**2 * np.exp(phi_dilaton / 4)
                    + 1.0 * r_arr * dphi_DE_dr * d_dilaton_dr
                    - 4.0 * dphi_DE_dr * d_dilaton_dr
                    - 8.0 * dphi_DE_dr / r_arr) / (4.0 * (r_arr - rs))

        return (np.array([dphi_DE_dr]) if np.isscalar(r) else np.asarray(dphi_DE_dr),  # ALWAYS return NumPy array
                np.array([ddphi_DE_dr2]) if np.isscalar(r) else np.asarray(ddphi_DE_dr2),
                np.array([dphi_DE_dt]) if np.isscalar(r) else np.asarray(dphi_DE_dt),
                np.array([ddphi_dt2]) if np.isscalar(r) else np.asarray(ddphi_dt2))
