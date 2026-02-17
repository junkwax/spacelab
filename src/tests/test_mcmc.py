import pytest
import numpy as np
from src.fitting.mcmc import (
    MCMCFitter,
    flat_log_prior,
    make_log_posterior,
    _StretchMoveSampler,
)


def _simple_log_prob(params):
    """2D Gaussian centered at (3.0, 5.0) with sigma=0.5."""
    x, y = params
    return -0.5 * ((x - 3.0) ** 2 + (y - 5.0) ** 2) / 0.5 ** 2


def test_flat_log_prior_inside():
    bounds = np.array([[0.0, 10.0], [0.0, 10.0]])
    assert flat_log_prior(np.array([5.0, 5.0]), bounds) == 0.0


def test_flat_log_prior_outside():
    bounds = np.array([[0.0, 10.0], [0.0, 10.0]])
    assert flat_log_prior(np.array([-1.0, 5.0]), bounds) == -np.inf


def test_make_log_posterior():
    bounds = np.array([[0.0, 10.0], [0.0, 10.0]])
    log_post = make_log_posterior(_simple_log_prob, lambda p: flat_log_prior(p, bounds))

    # Inside bounds: should return finite value
    assert np.isfinite(log_post(np.array([3.0, 5.0])))
    # Outside bounds: should return -inf
    assert log_post(np.array([-1.0, 5.0])) == -np.inf


def test_builtin_sampler_runs():
    """Built-in stretch-move sampler should run without errors."""
    sampler = _StretchMoveSampler(n_walkers=10, n_dim=2, log_prob_fn=_simple_log_prob)
    rng = np.random.default_rng(42)
    initial_pos = np.array([3.0, 5.0]) + 0.1 * rng.standard_normal((10, 2))

    sampler.run(initial_pos, n_steps=50)

    chain = sampler.get_chain()
    assert chain.shape == (50, 10, 2)

    log_probs = sampler.get_log_prob()
    assert log_probs.shape == (50, 10)


def test_mcmc_fitter_recovers_mean():
    """MCMCFitter should recover the center of a simple Gaussian."""
    fitter = MCMCFitter(
        log_posterior=_simple_log_prob,
        n_dim=2,
        n_walkers=16,
        param_names=["x", "y"],
    )
    fitter.run(
        initial_center=np.array([3.5, 4.5]),
        initial_spread=0.1,
        n_steps=500,
        progress=False,
    )

    summary = fitter.summary(discard=200, thin=2)
    assert abs(summary["x"]["median"] - 3.0) < 0.5
    assert abs(summary["y"]["median"] - 5.0) < 0.5


def test_gelman_rubin():
    """R-hat should be near 1 for a well-mixed chain."""
    fitter = MCMCFitter(log_posterior=_simple_log_prob, n_dim=2, n_walkers=16)
    fitter.run(
        initial_center=np.array([3.0, 5.0]),
        initial_spread=0.01,
        n_steps=300,
        progress=False,
    )

    r_hat = fitter.gelman_rubin(discard=100)
    assert r_hat.shape == (2,)
    # Should be reasonably close to 1
    assert np.all(r_hat < 2.0)


def test_acceptance_fraction():
    fitter = MCMCFitter(log_posterior=_simple_log_prob, n_dim=2, n_walkers=10)
    fitter.run(
        initial_center=np.array([3.0, 5.0]),
        initial_spread=0.01,
        n_steps=100,
        progress=False,
    )

    af = fitter.acceptance_fraction()
    assert 0.0 < af <= 1.0
