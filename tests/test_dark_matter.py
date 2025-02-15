import unittest
from models.dark_matter import DarkMatter

class TestDarkMatter(unittest.TestCase):
    def test_density_profile(self):
        dm = DarkMatter(mass=1e-22, coupling=1e-10)
        r = 1.0
        density = dm.density_profile(r)
        self.assertAlmostEqual(density, 1e-10 * np.exp(-1e-22) / 1.0)

if __name__ == "__main__":
    unittest.main()