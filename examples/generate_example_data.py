import numpy as np
import h5py

# Generate example simulation input
with h5py.File('examples/simulation_input.h5', 'w') as f:
    f.create_dataset('radius', data=np.linspace(0.1, 100, 256))
    f.create_dataset('mass', data=np.random.rand(256) * 1e6)

# Generate example fit input
with h5py.File('examples/fit_input.h5', 'w') as f:
    f.create_dataset('velocity', data=np.random.randn(100))
    f.create_dataset('error', data=np.random.rand(100) * 0.1)