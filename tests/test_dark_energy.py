# tests/test_dark_energy.py
import pytest
import numpy as np
from src.models.dark_energy import QuintessenceField

def test_quintessence_potential():
    q = QuintessenceField(V0=1e-10, lambda_=0.1, mass=1.0)  # Add mass
    assert np.isclose(q.potential(0), 1e-10)

def test_invalid_parameters():
    with pytest.raises(ValueError):
        QuintessenceField(V0=-1, lambda_=0.1, mass = 1.0)  # Include mass
    with pytest.raises(ValueError):
        QuintessenceField(V0=1e-10, lambda_=-0.1, mass = 1.0)  # Include mass

def test_equation_of_state():
    q = QuintessenceField(V0=1e-10, lambda_=0.1, mass=1.0)  # Add mass
    w = q.equation_of_state(0, 0)
    assert np.isclose(w, -1.0)

@pytest.fixture
def q():
    return QuintessenceField(V0=1e-47, lambda_=0.1, mass=1.0)  # Include mass

def test_field_equation_with_placeholders(q):
    y = (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))  # Example phi_DE, dphi_DE_dr, dphi_DE_dt
    r = np.array([3.0e9, 4.0e9, 5.0e9])
    phi_dilaton = np.array([0.0, 0.0, 0.0])
    phi_DM = np.array([0.1, 0.1, 0.1])  # Example phi_DM
    beta = 1e-10

    dphi_DE_dr, ddphi_DE_dr2, dphi_DE_dt, ddphi_DE_dt2 = q.field_equation(y, r, phi_dilaton, phi_DM, beta)

    assert isinstance(dphi_DE_dr, np.ndarray)
    assert isinstance(ddphi_DE_dr2, np.ndarray)
    assert isinstance(dphi_DE_dt, np.ndarray)
    assert isinstance(ddphi_DE_dt2, np.ndarray)
    assert dphi_DE_dr.shape == r.shape
    assert ddphi_DE_dr2.shape == r.shape
    assert dphi_DE_dt.shape == r.shape
    assert ddphi_DE_dt2.shape == r.shape

    # Test with a scalar 'r' value
    r_scalar = 3.0e9
    dphi_DE_dr_scalar, ddphi_DE_dr2_scalar, dphi_DE_dt_scalar, ddphi_DE_dt2_scalar = q.field_equation((np.array([0.0]), np.array([0.0]), np.array([0.0])), r_scalar, 0.0, 0.1, 1e-10)
    assert isinstance(dphi_DE_dr_scalar, np.ndarray)
    assert isinstance(ddphi_DE_dr2_scalar, np.ndarray)
    assert isinstance(dphi_DE_dt_scalar, np.ndarray)
    assert isinstance(ddphi_DE_dt2_scalar, np.ndarray)
    assert dphi_DE_dr_scalar.size == 1  # Check for correct shape
    assert ddphi_DE_dr2_scalar.size == 1
    assert dphi_DE_dt_scalar.size == 1
    assert ddphi_DE_dt2_scalar.size == 1

def test_field_equation_derivative_calculation(q):
    y = (np.array([0.1, 0.11, 0.12]), np.array([0.0,0.0,0.0]), np.array([0.02, 0.02, 0.02])) # phi_DE, dphi_DE_dr, dphi_DE_dt  Set initial dphi_DE_dr to zero
    r = np.array([3.0e9, 4.0e9, 5.0e9])
    dilaton = np.array([0.1, 0.2, 0.3])
    phi_DM = np.array([0.2, 0.2, 0.2])
    beta = 1e-10

    dphi_DE_dr, ddphi_DE_dr2, dphi_DE_dt, ddphi_DE_dt2 = q.field_equation(y, r, dilaton, phi_DM, beta)
    assert np.allclose(dphi_DE_dr, np.array([0.0, 0.0, 0.0]))
    assert not np.allclose(ddphi_DE_dr2, ddphi_DE_dr2[0], atol=1e-9) # Relaxed tolerance
    assert np.allclose(dphi_DE_dt, np.array([0.02, 0.02, 0.02]))
