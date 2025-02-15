import matplotlib.pyplot as plt

def plot_rotation_curve(radius, velocity, output_path):
    """Plot a galactic rotation curve."""
    plt.figure()
    plt.plot(radius, velocity, label='Predicted Velocity')
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Velocity (km/s)')
    plt.legend()
    plt.savefig(output_path)
    plt.close()