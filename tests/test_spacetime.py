import unittest
import numpy as np
from src.models.spacetime import Spacetime, schwarzschild_metric, ricci_curvature

class TestSpacetime(unittest.TestCase):
    def test_metric(self):
        r = 10.0
        metric = schwarzschild_metric(r)
        self.assertEqual(metric.shape, (4, 4))

    def test_curvature(self):
        r = 10.0
        self.assertEqual(ricci_curvature(r), 0)

if __name__ == "__main__":
    unittest.main()