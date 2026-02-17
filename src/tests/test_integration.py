import pytest
import yaml
import numpy as np
from src.simulation import run_simulation


def test_black_hole_simulation(tmp_path):
    """Integration test for the black hole simulation mode."""
    config = {
        "mode": "black_hole",
        "dark_matter": {"mass": 1e-22, "coupling_dilaton": 1e-10, "coupling_curvature": 1e-5},
        "dark_energy": {"V0": 1e-47, "lambda_": 0.1},
        "spacetime": {"black_hole_mass": 1e6},
        "fields": {"dilaton": 1.0, "graviphoton": 0.0},
        "numerical": {"grid_size": 32},
    }

    config_file = tmp_path / "test_bh.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f)

    solution = run_simulation(str(config_file))
    assert isinstance(solution, np.ndarray)
    assert solution.shape == (32, 4)
    assert np.all(np.isfinite(solution))


def test_cosmology_simulation(tmp_path):
    """Integration test for the cosmological evolution mode."""
    config = {
        "mode": "cosmology",
        "dark_matter": {"mass": 1e-22, "coupling_dilaton": 1e-10, "coupling_curvature": 1e-5},
        "dark_energy": {"V0": 1e-47, "lambda_": 0.1},
        "spacetime": {"black_hole_mass": 1e6},
        "cosmology": {
            "H0": 67.4, "Omega_m": 0.315, "Omega_r": 9.1e-5,
            "phi_0": 1.0, "dphi_dt_0": 0.0,
            "a_start": 0.001, "a_end": 1.0,
        },
        "numerical": {"grid_size": 50},
    }

    config_file = tmp_path / "test_cosmo.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f)

    result = run_simulation(str(config_file))
    assert isinstance(result, dict)
    assert "a" in result and "phi" in result and "w_DE" in result
    assert len(result["a"]) == 50
    assert np.all(np.isfinite(result["phi"]))
