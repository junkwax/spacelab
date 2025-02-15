import unittest
import numpy as np
from src.models.dark_energy import DarkEnergy

class TestDarkEnergy(unittest.TestCase):
    def test_potential(self):
        de = DarkEnergy(V0=1e-10, lambda_=0.1)
        self.assertAlmostEqual(de.potential(0), 1e-10)

    def test_equation_of_state(self):
        de = DarkEnergy(V0=1e-10, lambda_=0.1)
        w = de.equation_of_state(phi=0, dphi_dt=0)
        self.assertAlmostEqual(w, -1.0)  # Cosmological constant

if __name__ == "__main__":
    unittest.main()