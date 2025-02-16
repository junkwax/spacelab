# tests/test_integration.py

import pytest
import yaml
import numpy as np
from src.simulation import run_simulation, einstein_field_equations  # Import einstein_field_equations
from src.models.spacetime import SpacetimeGeometry

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
        yaml.safe_dump(config, f)
    
    # Run simulation
    try:
        run_simulation(config_file)
    except Exception as e:
        pytest.fail(f"Simulation failed with error: {str(e)}")

def test_einstein_field_equations_schwarzschild():
    """Tests the Einstein field equation solver with a zero stress-energy tensor (Schwarzschild)."""
    spacetime = SpacetimeGeometry(mass=1) # Solar Mass
    r = np.linspace(3, 10, 100)
    T_munu = np.zeros((4, 4, len(r)))  # Zero stress-energy tensor

    A, B = einstein_field_equations(r, T_munu, spacetime)

    assert isinstance(A, np.ndarray)
    assert isinstance(B, np.ndarray)
    assert A.shape == r.shape
    assert B.shape == r.shape

    # Check if the solution is close to Schwarzschild
    mass_kg = spacetime.mass * spacetime.solar_mass_kg
    rs = 2 * spacetime.G * mass_kg / (spacetime.c ** 2)
    expected_A = 1 / (1 - rs / r)
    expected_B = -(1 - rs / r)

    assert np.allclose(A, expected_A, atol=1e-4)  # Relax tolerance
    assert np.allclose(B, expected_B, atol=1e-4)
