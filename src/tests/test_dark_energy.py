import pytest
import numpy as np
from src.models.dark_energy import QuintessenceField


@pytest.fixture
def quintessence():
    return QuintessenceField(V0=1e-47, lambda_=0.1)


def test_potential_at_zero(quintessence):
    """V(0) should equal V0."""
    assert np.isclose(quintessence.potential(0.0), 1e-47)


def test_potential_decreases(quintessence):
    """Potential should decrease with increasing phi for positive lambda."""
    v0 = quintessence.potential(0.0)
    v1 = quintessence.potential(1.0)
    assert v1 < v0


def test_potential_vectorized(quintessence):
    phi = np.array([0.0, 1.0, 2.0])
    result = quintessence.potential(phi)
    assert result.shape == (3,)
    assert np.all(np.diff(result) < 0)  # monotonically decreasing


def test_equation_of_state_static_field(quintessence):
    """A static field (dphi_dt=0) should give w = -1 (cosmological constant)."""
    w = quintessence.equation_of_state(phi=1.0, dphi_dt=0.0)
    assert np.isclose(w, -1.0)


def test_equation_of_state_kinetic_dominated(quintessence):
    """When kinetic >> potential, w should approach +1."""
    w = quintessence.equation_of_state(phi=0.0, dphi_dt=1e10)
    assert w > 0.99


def test_equation_of_state_zero_denominator(quintessence):
    """When both K and V are zero, should return -1."""
    qf = QuintessenceField(V0=1.0, lambda_=1.0)
    # V(phi) = exp(-phi) → for very large phi, V ≈ 0, and dphi_dt = 0 → K = 0
    # Use explicit zero
    w = qf.equation_of_state(phi=1000.0, dphi_dt=0.0)
    # V is effectively 0, K is 0 → denominator is 0
    assert np.isclose(w, -1.0)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        QuintessenceField(V0=-1.0, lambda_=0.1)
    with pytest.raises(ValueError):
        QuintessenceField(V0=1.0, lambda_=-0.1)


def test_field_acceleration(quintessence):
    """Basic sanity: acceleration should be finite for reasonable inputs."""
    acc = quintessence.field_acceleration(phi=0.1, dphi_dt=0.01, H=70.0)
    assert np.isfinite(acc)
