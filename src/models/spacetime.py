import numpy as np

class Spacetime:
    def __init__(self, metric_func, curvature_func):
        self.metric_func = metric_func
        self.curvature_func = curvature_func

    def metric(self, r):
        """Compute the spacetime metric tensor at radius r."""
        return self.metric_func(r)

    def curvature(self, r):
        """Compute the Ricci scalar curvature at radius r."""
        return self.curvature_func(r)

# Example metric and curvature functions
def schwarzschild_metric(r, M=1e6):
    """Schwarzschild metric in spherical coordinates."""
    return np.diag([-(1 - 2*M/r), 1/(1 - 2*M/r), r**2, r**2*np.sin(np.pi/2)**2])

def ricci_curvature(r, M=1e6):
    """Ricci scalar curvature for Schwarzschild spacetime."""
    return 0  # Schwarzschild is Ricci-flat