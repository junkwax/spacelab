"""
MCMC parameter fitting for SpaceLab models.

Provides two backends:
  1. Built-in Metropolis-Hastings (no extra dependencies).
  2. emcee ensemble sampler (if installed).

Typical usage:
    sampler = MCMCSampler(log_posterior_fn, param_names, priors, ...)
    chain = sampler.run(n_steps=5000)
    sampler.summary()
"""

import numpy as np
from typing import Callable, Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class MCMCSampler:
    """Lightweight MCMC wrapper.

    Args:
        log_prob_fn: Callable(params) -> float.  The log-posterior.
        param_names: List of parameter names.
        priors: Dict of param_name -> (low, high) bounds.
        n_walkers: Number of walkers (only used with emcee backend).
        backend: 'metropolis' or 'emcee'.
    """

    def __init__(
        self,
        log_prob_fn: Callable,
        param_names: List[str],
        priors: Dict[str, Tuple[float, float]],
        n_walkers: int = 32,
        backend: str = "metropolis",
    ):
        self.log_prob_fn = log_prob_fn
        self.param_names = param_names
        self.priors = priors
        self.ndim = len(param_names)
        self.n_walkers = n_walkers
        self.backend = backend

        self.chain = None           # shape: (n_steps, ndim)
        self.log_prob_chain = None  # shape: (n_steps,)
        self.acceptance_rate = 0.0

    def _initial_guess(self, seed: int = 42) -> np.ndarray:
        """Draw an initial point uniformly within priors."""
        rng = np.random.default_rng(seed)
        p0 = np.zeros(self.ndim)
        for i, name in enumerate(self.param_names):
            lo, hi = self.priors[name]
            p0[i] = rng.uniform(lo, hi)
        return p0

    def _initial_ball(self, center: np.ndarray, spread: float = 0.01, seed: int = 42) -> np.ndarray:
        """Generate an (n_walkers, ndim) ball around center for emcee."""
        rng = np.random.default_rng(seed)
        ball = center + spread * rng.standard_normal((self.n_walkers, self.ndim))
        # Clip to priors
        for i, name in enumerate(self.param_names):
            lo, hi = self.priors[name]
            ball[:, i] = np.clip(ball[:, i], lo, hi)
        return ball

    # ------------------------------------------------------------------
    # Metropolis-Hastings
    # ------------------------------------------------------------------

    def _run_metropolis(
        self,
        n_steps: int,
        p0: Optional[np.ndarray],
        proposal_scale: float,
        seed: int,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)

        if p0 is None:
            p0 = self._initial_guess(seed)

        current = p0.copy()
        current_lp = self.log_prob_fn(current)

        chain = np.zeros((n_steps, self.ndim))
        lp_chain = np.zeros(n_steps)
        n_accept = 0

        # Adaptive proposal widths based on prior range
        widths = np.array([
            (self.priors[name][1] - self.priors[name][0]) * proposal_scale
            for name in self.param_names
        ])

        for step in range(n_steps):
            proposal = current + rng.normal(0, widths)

            # Reflect off prior boundaries
            for i, name in enumerate(self.param_names):
                lo, hi = self.priors[name]
                proposal[i] = np.clip(proposal[i], lo, hi)

            proposal_lp = self.log_prob_fn(proposal)

            log_alpha = proposal_lp - current_lp
            if np.log(rng.uniform()) < log_alpha:
                current = proposal
                current_lp = proposal_lp
                n_accept += 1

            chain[step] = current
            lp_chain[step] = current_lp

            if (step + 1) % 1000 == 0:
                logger.info(
                    f"  Step {step+1}/{n_steps}, "
                    f"accept rate={n_accept/(step+1):.3f}, "
                    f"log_prob={current_lp:.2f}"
                )

        self.acceptance_rate = n_accept / n_steps
        return chain, lp_chain

    # ------------------------------------------------------------------
    # emcee backend
    # ------------------------------------------------------------------

    def _run_emcee(
        self,
        n_steps: int,
        p0: Optional[np.ndarray],
        seed: int,
    ) -> np.ndarray:
        try:
            import emcee
        except ImportError:
            raise ImportError(
                "emcee is not installed. Install with `pip install emcee` "
                "or use backend='metropolis'."
            )

        if p0 is None:
            center = self._initial_guess(seed)
        else:
            center = p0

        pos = self._initial_ball(center, seed=seed)

        sampler = emcee.EnsembleSampler(
            self.n_walkers, self.ndim, self.log_prob_fn
        )
        sampler.run_mcmc(pos, n_steps, progress=True)

        self.acceptance_rate = np.mean(sampler.acceptance_fraction)

        # Flatten chains: (n_walkers * n_steps, ndim)
        chain = sampler.get_chain(flat=True)
        lp_chain = sampler.get_log_prob(flat=True)
        return chain, lp_chain

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        n_steps: int = 5000,
        p0: Optional[np.ndarray] = None,
        proposal_scale: float = 0.02,
        burn_in: int = 1000,
        seed: int = 42,
    ) -> np.ndarray:
        """Run the MCMC sampler.

        Args:
            n_steps: Total number of steps (including burn-in).
            p0: Initial parameter vector. If None, drawn from priors.
            proposal_scale: Fraction of prior width for MH proposals.
            burn_in: Number of initial steps to discard.
            seed: Random seed.

        Returns:
            Chain array of shape (n_kept, ndim).
        """
        logger.info(
            f"Running MCMC ({self.backend}): {n_steps} steps, "
            f"{self.ndim} parameters, burn_in={burn_in}"
        )

        if self.backend == "metropolis":
            chain, lp_chain = self._run_metropolis(n_steps, p0, proposal_scale, seed)
        elif self.backend == "emcee":
            chain, lp_chain = self._run_emcee(n_steps, p0, seed)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        # Discard burn-in
        self.chain = chain[burn_in:]
        self.log_prob_chain = lp_chain[burn_in:]

        logger.info(
            f"MCMC complete. Acceptance rate: {self.acceptance_rate:.3f}, "
            f"chain shape: {self.chain.shape}"
        )

        return self.chain

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Print and return parameter summary (median, 16th, 84th percentiles)."""
        if self.chain is None:
            raise RuntimeError("No chain available. Call run() first.")

        results = {}
        print(f"\n{'Parameter':<20} {'Median':>12} {'−1σ':>12} {'+1σ':>12}")
        print("-" * 60)

        for i, name in enumerate(self.param_names):
            samples = self.chain[:, i]
            q16, q50, q84 = np.percentile(samples, [16, 50, 84])
            results[name] = {
                "median": q50,
                "lower": q50 - q16,
                "upper": q84 - q50,
            }
            print(f"{name:<20} {q50:>12.4e} {q50-q16:>12.4e} {q84-q50:>12.4e}")

        # Best-fit (MAP)
        best_idx = np.argmax(self.log_prob_chain)
        best_params = self.chain[best_idx]
        print(f"\nBest-fit (MAP): log_prob = {self.log_prob_chain[best_idx]:.2f}")
        for name, val in zip(self.param_names, best_params):
            print(f"  {name} = {val:.4e}")

        results["_best_fit"] = dict(zip(self.param_names, best_params))
        results["_best_log_prob"] = float(self.log_prob_chain[best_idx])

        return results

    def get_best_fit(self) -> np.ndarray:
        """Return the MAP (maximum a posteriori) parameter vector."""
        if self.chain is None:
            raise RuntimeError("No chain available. Call run() first.")
        best_idx = np.argmax(self.log_prob_chain)
        return self.chain[best_idx]
