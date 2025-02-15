import yaml
import h5py
from models.dark_matter import DarkMatter
from models.dark_energy import DarkEnergy
from utils.data_loader import load_data
from utils.plotting import plot_results

def run_simulation(config_file):
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize models
    dark_matter = DarkMatter(config['dark_matter']['mass'], config['dark_matter']['coupling'])
    dark_energy = DarkEnergy(config['dark_energy']['V0'], config['dark_energy']['lambda_'])

    # Run simulation (placeholder for actual numerical solver)
    results = {
        'radius': np.linspace(0.1, 100, config['numerical']['grid_size']),
        'density': dark_matter.density_profile(np.linspace(0.1, 100, config['numerical']['grid_size']))
    }

    # Save results
    with h5py.File('results/simulation_results.h5', 'w') as f:
        for key, value in results.items():
            f.create_dataset(key, data=value)

    # Plot results
    plot_results(results)

if __name__ == "__main__":
    run_simulation('configs/simulation_config.yaml')