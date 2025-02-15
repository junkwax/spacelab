import h5py

def load_h5_data(filepath):
    """Load data from an HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        data = {key: f[key][()] for key in f.keys()}
    return data

def save_h5_data(filepath, data_dict):
    """Save data to an HDF5 file."""
    with h5py.File(filepath, 'w') as f:
        for key, value in data_dict.items():
            f.create_dataset(key, data=value)