import matplotlib.pyplot as plt

def plot_results(results):
    """Plot simulation results."""
    plt.figure()
    plt.plot(results['radius'], results['density'], label='Dark Matter Density')
    plt.xlabel('Radius')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('results/plots/density_profile.png')
    plt.close()