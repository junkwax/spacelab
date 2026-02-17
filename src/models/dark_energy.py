import numpy as np
from typing import Union
import logging

logger = logging.getLogger(__name__)


class QuintessenceField:
    """Model for dynamical dark energy via a quintessence scalar field.

    The potential is an exponential run-away form:  V(φ) = V0 * exp(-λ φ).

    Args:
        V0: Amplitude of the potential [GeV^4].
        lambda_: Slope parameter (positive).
    """

    def __init__(self, V0: float, lambda_: float):
        if V0 <= 0 or lambda_ <= 0:
            raise ValueError("V0 and lambda must be positive.")
        self.V0 = V0
        self.lambda_ = lambda_

    def potential(self, phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate the quintessence potential V(φ)."""
        return self.V0 * np.exp(-self.lambda_ * phi)

    def equation_of_state(
        self,
        phi: Union[float, np.ndarray],
        dphi_dt: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute the dark energy equation-of-state parameter w = (K - V) / (K + V).

        Returns w = -1 when the denominator is zero (pure cosmological constant limit).
        """
        kinetic = 0.5 * dphi_dt ** 2
        potential = self.potential(phi)
        denominator = kinetic + potential

        if np.any(denominator == 0):
            return np.full_like(np.asarray(denominator, dtype=float), -1.0)

        return (kinetic - potential) / denominator

    def field_acceleration(
        self,
        phi: Union[float, np.ndarray],
        dphi_dt: Union[float, np.ndarray],
        H: float,
    ) -> Union[float, np.ndarray]:
        """Compute d²φ/dt² for the Klein-Gordon equation in an FRW background.

        d²φ/dt² = -3H dφ/dt - dV/dφ

        Args:
            phi: Current field value.
            dphi_dt: Current time derivative of the field.
            H: Hubble parameter.

        Returns:
            Second time derivative of the field.
        """
        dV_dphi = -self.lambda_ * self.V0 * np.exp(-self.lambda_ * phi)
        return -3.0 * H * dphi_dt - dV_dphi
