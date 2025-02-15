from pydantic import BaseModel, PositiveFloat

class SimulationConfig(BaseModel):
    black_hole_mass: PositiveFloat
    black_hole_spin: float = 0.7  # Allow 0 ≤ spin < 1
    dark_matter_mass: PositiveFloat
    dark_matter_coupling: PositiveFloat

def validate_config(config_data: dict) -> SimulationConfig:
    """Validate simulation configuration using Pydantic."""
    return SimulationConfig(**config_data)