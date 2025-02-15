import pytest
from src.black_hole_simulation import run_simulation

def test_simulation_workflow(tmp_path):
    config = {
        "black_hole": {"mass": 1e6, "spin": 0.7},
        "dark_matter": {"mass": 1e-22, "coupling": 1e-10}
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f)
    
    # Ensure the simulation runs without errors
    run_simulation(config_file)