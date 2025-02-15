import h5py
import matplotlib.pyplot as plt

def analyze_fit_results(input_file, plot_dir):
    """Analyze MCMC fit results."""
    with h5py.File(input_file, 'r') as f:
        samples = f['samples'][()]
        
    plt.figure()
    plt.hist(samples[:, :, 0].flatten(), bins=50, label='Mass')
    plt.hist(samples[:, :, 1].flatten(), bins=50, label='Coupling')
    plt.legend()
    plt.savefig(f"{plot_dir}/parameter_distributions.png")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--plot', type=str, required=True)
    args = parser.parse_args()
    analyze_fit_results(args.input, args.plot)