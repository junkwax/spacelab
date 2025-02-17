# src/models/dark_matter.py
import numpy as np
from typing import Union, overload, Tuple

class DarkMatter:
    """Model for axion-like dark matter in higher-dimensional spacetime."""
    def __init__(self, mass: float, coupling_dilaton: float, coupling_curvature: float):
        if mass <= 0 or coupling_dilaton <= 0 or coupling_curvature <= 0:
            raise ValueError("Mass and coupling constants must be positive.")
        self.mass = mass
        self.coupling_dilaton = coupling_dilaton
        self.coupling_curvature = coupling_curvature
        self.solar_mass_kg = 1.98847e30
        self.G = 6.67430e-11  # Gravitational constant [m^3 kg^-1 s^-2]
        self.c = 299792458  # Speed of light [m/s]

    @overload
    def density_profile(self, r: float, dilaton_field: float) -> float: ...

    @overload
    def density_profile(self, r: np.ndarray, dilaton_field: np.ndarray) -> np.ndarray: ...

    def density_profile(self, r: Union[float, np.ndarray], dilaton_field: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the dark matter density profile."""
        r_arr = np.asarray(r)
        if np.any(r_arr <= 0):
            raise ValueError("Radius `r` must be positive.")
        return self.coupling_dilaton * np.exp(-self.mass * r_arr) * dilaton_field / r_arr**2  # Placeholder

    def potential(self, phi: Union[float, np.ndarray], r: Union[float, np.ndarray], dilaton_field: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Scalar field potential."""
        R = 2.0 / r**2  # Placeholder
        return 0.5 * self.mass**2 * phi**2 + self.coupling_curvature * R * phi**2 + self.coupling_dilaton * dilaton_field * phi**2

    def field_equation(self, y: Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], r: Union[float, np.ndarray], dilaton_field: Union[float, np.ndarray], graviphoton_field: Union[float, np.ndarray], phi_DE: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Scalar field equation."""
        phi_DM, dphi_DM_dr = y
        r_arr = np.atleast_1d(np.asarray(r))
        phi_DM = np.atleast_1d(np.asarray(phi_DM))
        dphi_DM_dr = np.atleast_1d(np.asarray(dphi_DM_dr))
        dilaton_field = np.atleast_1d(np.asarray(dilaton_field))
        phi_DE = np.atleast_1d(np.asarray(phi_DE))
        graviphoton_field = np.atleast_1d(np.asarray(graviphoton_field)) # Make graviphoton_field array compatible

        # Calculate derivatives of dilaton_field
        if r_arr.size <= 1: # Handle scalar r case
            d_dilaton_dr = np.array([0.0])
            d_graviphoton_dr = np.array([0.0]) # Derivative is zero for scalar r
        else:
            d_dilaton_dr = np.gradient(dilaton_field, r_arr)
            d_graviphoton_dr = np.gradient(graviphoton_field, r_arr)


        # Convert mass from solar masses to kilograms and calculate rs
        mass_kg = self.mass * self.solar_mass_kg
        rs = 2 * self.G * mass_kg / (self.c**2)

        # Ensure radius is greater than Schwarzschild radius
        if (r_arr <= rs).any():
            raise ValueError("Radius must be greater than the Schwarzschild radius.")

        # Calculate ddphi_dr2
        ddphi_dr2 = (
            2.0 * r_arr * phi_DM * self.coupling_dilaton * np.exp(dilaton_field / 4)
            + 1.0 * r_arr * phi_DM * self.mass**2 * np.exp(dilaton_field / 4)
            + 2.0 * r_arr * dphi_DM_dr * d_dilaton_dr
            - 8.0 * phi_DM * self.coupling_dilaton * np.exp(dilaton_field / 4)
            - 4.0 * phi_DM * self.mass**2 * np.exp(dilaton_field / 4)
            - 4.0 * dphi_DM_dr * d_dilaton_dr
            - 8.0 * dphi_DM_dr / r_arr
        ) / (4.0 * (r_arr - rs))

        return (dphi_DM_dr, ddphi_dr2)
