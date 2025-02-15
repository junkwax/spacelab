import emcee
import h5py
import numpy as np
from src.models.dark_matter import DarkMatter

def log_likelihood(theta, data):
    """Log-likelihood function for MCMC."""
    mass, coupling = theta
    dm = DarkMatter(mass=mass, coupling=coupling)
    model = dm.density_profile(data['radius'])
    residuals = data['velocity'] - model
    return -0.5 * np.sum(residuals**2 / data['error']**2)

def run_global_fits(data_file, output_file):
    """Run MCMC global fits."""
    data = h5py.File(data_file, 'r')
    sampler = emcee.EnsembleSampler(32, 2, log_likelihood, args=[data])
    pos = np.random.randn(32, 2) * [1e-22, 1e-10]
    sampler.run_mcmc(pos, 1000)
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('samples', data=sampler.get_chain())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    run_global_fits(args.data, args.output)