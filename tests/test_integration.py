import pytest
import yaml
from src.simulation import run_simulation

def test_simulation_workflow(tmp_path):
    """Integration test for the full simulation workflow."""
        
    # Create test config
    config = {
        "black_hole": {
            "mass": 1e6,  # Solar masses
            "spin": 0.7
        },
        "dark_matter": {
            "mass": 1e-22,  # eV
            "coupling_dilaton": 1e-10, 
            "coupling_curvature": 1e-5  
        },
        "dark_energy": { 
            "V0": 1e-47,  # Example value for V0
            "lambda_": 0.1  # Example value for lambda_
        },
        "spacetime": {  # Add the spacetime section
            "black_hole_mass": 1e6  # Example value for black hole mass
        },
        "numerical": {
            "grid_size": 128,
            "time_steps": 500
        }
    }

    
    # Write config to temporary file
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f)  # Now has yaml imported
    
    # Run simulation
    try:
        run_simulation(config_file)
    except Exception as e:
        pytest.fail(f"Simulation failed with error: {str(e)}")