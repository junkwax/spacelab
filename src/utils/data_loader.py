import h5py

class DataLoadError(Exception):
    """Custom exception for data loading errors."""

def load_h5_data(filepath):
    """Load data from an HDF5 file.
    
    Args:
        filepath (str): Path to the HDF5 file.
    
    Returns:
        dict: Dictionary containing datasets.
    
    Raises:
        DataLoadError: If the file cannot be loaded.
    """
    try:
        with h5py.File(filepath, 'r') as f:
            data = {key: f[key][()] for key in f.keys()}
        return data
    except FileNotFoundError:
        raise DataLoadError(f"File {filepath} not found.")
    except Exception as e:
        raise DataLoadError(f"Error loading {filepath}: {str(e)}")