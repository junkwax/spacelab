import numpy as np
from typing import Union, overload
import logging

logger = logging.getLogger(__name__)

class QuintessenceField:
    """Model for dynamical dark energy (quintessence) with a scalar field.

    Args:
        V0 (float): Energy scale of the potential [eV^4].
        lambda_ (float): Slope parameter of the exponential potential.
    """
    def __init__(self, V0: float, lambda_: float):
        if V0 <= 0 or lambda_ <= 0:
            logger.error("Invalid quintessence parameters: V0=%s, lambda=%s", V0, lambda_)
            raise ValueError("V0 and lambda must be positive.")
        self.V0 = V0
        self.lambda_ = lambda_
        self.solar_mass_kg = 1.98847e30
        self.G = 6.67430e-11
        self.c = 299792458


    def potential(self, phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the quintessence potential V(φ).

        Args:
            phi (float or np.ndarray): Scalar field value [eV]

        Returns:
            Potential energy [eV^4]
        """
        return self.V0 * np.exp(-self.lambda_ * phi)

    def equation_of_state(
        self,
        phi: Union[float, np.ndarray],
        dphi_dt: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Compute the equation of state parameter w = P/ρ.

        Args:
            phi: Scalar field value [eV]
            dphi_dt: Time derivative of phi [eV^2]

        Returns:
            Equation of state parameter (dimensionless)
        """
        kinetic = 0.5 * dphi_dt**2
        potential = self.potential(phi)
        denominator = kinetic + potential

        if np.any(denominator == 0):
            logger.warning("Division by zero in equation_of_state calculation")
            return np.full_like(denominator, -1.0)  # Return w=-1 for stability

        return (kinetic - potential) / denominator

    def field_equation(self, y, r, phi_dilaton, phi_DM, beta):
        """Quintessence field equation.

        Args:
            y (tuple): (phi_DE, dphi_DE_dr, dphi_DE_dt)
            r (float or array-like): Radial coordinate.
            phi_dilaton (float or array-like): Dilaton field value.
            phi_DM (float or array-like): Dark matter field value.
            beta (float): Coupling constant between dark matter and dark energy.

        Returns:
            list: [dphi_DE_dr, ddphi_DE_dr2, dphi_DE_dt, ddphi_DE_dt2]
        """
        phi_DE, dphi_DE_dr, dphi_DE_dt = y

        r = np.asarray(r)
        phi_dilaton = np.asarray(phi_dilaton)
        phi_DM = np.asarray(phi_DM)

        # Calculate derivatives
        d_dilaton_dr = np.gradient(phi_dilaton, r)

        # Convert mass from solar masses to kilograms and calculate rs
        mass_kg = self.mass * self.solar_mass_kg
        rs = 2 * self.G * mass_kg / (self.c**2)

        # Ensure radius is greater than Schwarzschild radius
        if np.any(r <= rs):
            raise ValueError("Radius must be greater than the Schwarzschild radius.")

        # Calculate ddphi_dt2 and ddphi_dr2 using the derived expressions
        ddphi_dt2 = (1.0 * self.V0 * self.lambda_ * r**2 * np.exp(phi_dilaton / 4) * np.exp(-self.lambda_ * phi_DE)
                    + 1.0 * beta * r**2 * phi_DM**2 * np.exp(phi_dilaton / 4)
                    + 1.0 * dphi_DE_dt * d_dilaton_dr) / (r * (r - rs))

        ddphi_dr2 = (-2.0 * self.V0 * self.lambda_ * r**2 * np.exp(phi_dilaton / 4) * np.exp(-self.lambda_ * phi_DE)
                    - 2.0 * beta * r**2 * phi_DM**2 * np.exp(phi_dilaton / 4)
                    + 8.0 * self.V0 * self.lambda_ * r * np.exp(phi_dilaton / 4) * np.exp(-self.lambda_ * phi_DE)
                    + 8.0 * beta * r * phi_DM**2 * np.exp(phi_dilaton / 4)
                    + 1.0 * r * dphi_DE_dr * d_dilaton_dr
                    - 4.0 * dphi_DE_dr * d_dilaton_dr
                    - 8.0 * dphi_DE_dr / r) / (4.0 * (r - rs))

        return [dphi_DE_dr, ddphi_dr2, dphi_DE_dt, ddphi_dt2]
