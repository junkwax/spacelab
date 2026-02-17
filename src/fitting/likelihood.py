"""
Likelihood functions for fitting SpaceLab models to observational data.

All velocities in km/s, all radii in kpc (matching SPARC convention).
"""

import numpy as np
from typing import Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def rotation_curve_chi2(
    params: np.ndarray,
    data: Dict[str, np.ndarray],
    model_fn: Callable,
    param_names: list,
    priors: Optional[Dict[str, tuple]] = None,
) -> float:
    """Compute χ² for a rotation curve model vs data.

    Args:
        params: Array of parameter values.
        data: Dict with 'r_kpc', 'v_obs_kms', 'v_err_kms'.
        model_fn: Callable(r_kpc, **params) -> v_total_kms.
        param_names: Names corresponding to params array.
        priors: Optional dict of (min, max) bounds per parameter.

    Returns:
        χ² value (1e30 for out-of-prior or failed evaluations).
    """
    param_dict = dict(zip(param_names, params))

    # Check priors (flat / uniform)
    if priors is not None:
        for name, (lo, hi) in priors.items():
            if name in param_dict:
                if param_dict[name] < lo or param_dict[name] > hi:
                    return 1e30

    try:
        v_model = model_fn(data["r_kpc"], **param_dict)
    except Exception as e:
        logger.debug(f"Model evaluation failed: {e}")
        return 1e30

    residuals = (data["v_obs_kms"] - v_model) / data["v_err_kms"]
    return float(np.sum(residuals**2))


def log_likelihood(
    params: np.ndarray,
    data: Dict[str, np.ndarray],
    model_fn: Callable,
    param_names: list,
    priors: Optional[Dict[str, tuple]] = None,
) -> float:
    """Log-likelihood for Gaussian errors: ln L = -χ²/2."""
    chi2 = rotation_curve_chi2(params, data, model_fn, param_names, priors)
    if chi2 >= 1e29:
        return -np.inf
    return -0.5 * chi2


def log_prior_flat(
    params: np.ndarray,
    param_names: list,
    priors: Dict[str, tuple],
) -> float:
    """Flat (uniform) prior. Returns 0 if in bounds, -∞ otherwise."""
    for name, val in zip(param_names, params):
        if name in priors:
            lo, hi = priors[name]
            if val < lo or val > hi:
                return -np.inf
    return 0.0


def log_posterior(
    params: np.ndarray,
    data: Dict[str, np.ndarray],
    model_fn: Callable,
    param_names: list,
    priors: Dict[str, tuple],
) -> float:
    """Log-posterior = log-prior + log-likelihood."""
    lp = log_prior_flat(params, param_names, priors)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, data, model_fn, param_names, priors=None)
    return lp + ll


def reduced_chi2(
    params: np.ndarray,
    data: Dict[str, np.ndarray],
    model_fn: Callable,
    param_names: list,
) -> float:
    """χ²/dof — useful for goodness-of-fit assessment."""
    chi2 = rotation_curve_chi2(params, data, model_fn, param_names)
    dof = len(data["r_kpc"]) - len(param_names)
    return chi2 / max(dof, 1)
