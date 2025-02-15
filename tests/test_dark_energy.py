# tests/test_dark_energy.py
import pytest
import numpy as np
from src.models.dark_energy import QuintessenceField

def test_quintessence_potential():
    q = QuintessenceField(V0=1e-10, lambda_=0.1)
    assert np.isclose(q.potential(0), 1e-10)

def test_invalid_parameters():
    with pytest.raises(ValueError):
        QuintessenceField(V0=-1, lambda_=0.1)
    with pytest.raises(ValueError):
        QuintessenceField(V0=1e-10, lambda_=-0.1)

def test_equation_of_state():
    q = QuintessenceField(V0=1e-10, lambda_=0.1)
    w = q.equation_of_state(0, 0)
    assert np.isclose(w, -1.0)