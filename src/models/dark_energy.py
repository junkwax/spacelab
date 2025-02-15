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