"""Plotting utilities for SpaceLab simulations.

All functions save figures to an output directory and return the file path.
A non-interactive matplotlib backend is used so the code works in headless
environments (CI, HPC).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "output"


def _ensure_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)


# -----------------------------------------------------------------------
# Black hole scalar field
# -----------------------------------------------------------------------

def plot_scalar_field(
    r: np.ndarray,
    phi: np.ndarray,
    dphi_dr: np.ndarray,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    """Plot the dark-matter scalar field and its radial derivative.

    Returns:
        Path to the saved image.
    """
    _ensure_dir(output_dir)
    filename = os.path.join(output_dir, "scalar_field_profile.png")
    logger.info(f"Generating plot at {filename}...")

    try:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel("Radius $r$ (m)")
        ax1.set_ylabel(r"Scalar Field $\phi$", color="tab:blue")
        ax1.plot(r, phi, color="tab:blue", linewidth=2, label=r"$\phi$")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel(r"Derivative $d\phi/dr$", color="tab:red")
        ax2.plot(r, dphi_dr, color="tab:red", linestyle="--", linewidth=1.5)
        ax2.tick_params(axis="y", labelcolor="tab:red")

        plt.title("Dark Matter Scalar Field Profile around Black Hole")
        fig.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        logger.info("Plot generated successfully.")
        return filename

    except Exception as e:
        logger.error(f"Failed to generate plot: {e}")
        raise RuntimeError(f"Plot generation failed: {e}") from e


# -----------------------------------------------------------------------
# Cosmological evolution
# -----------------------------------------------------------------------

def plot_cosmology(
    result: Dict[str, np.ndarray],
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    """Plot quintessence evolution: phi(a), w_DE(a), H(a).

    Args:
        result: Dictionary from FRWCosmology.evolve_quintessence().

    Returns:
        Path to the saved image.
    """
    _ensure_dir(output_dir)
    filename = os.path.join(output_dir, "cosmology_evolution.png")
    logger.info(f"Generating cosmology plot at {filename}...")

    try:
        a = result["a"]
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Panel 1: scalar field
        axes[0].plot(a, result["phi"], color="tab:blue", linewidth=2)
        axes[0].set_ylabel(r"Quintessence field $\phi$")
        axes[0].grid(True, alpha=0.3)

        # Panel 2: equation of state
        axes[1].plot(a, result["w_DE"], color="tab:orange", linewidth=2)
        axes[1].axhline(-1.0, color="gray", linestyle="--", alpha=0.5, label=r"$\Lambda$CDM ($w=-1$)")
        axes[1].set_ylabel(r"Equation of state $w_\mathrm{DE}$")
        axes[1].set_ylim(-1.5, 0.5)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Panel 3: Hubble parameter
        axes[2].plot(a, result["H"], color="tab:green", linewidth=2)
        axes[2].set_ylabel(r"$H(a)$ [Gyr$^{-1}$]")
        axes[2].set_xlabel("Scale factor $a$")
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("Quintessence Dark Energy Evolution", fontsize=14)
        fig.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        logger.info("Cosmology plot generated successfully.")
        return filename

    except Exception as e:
        logger.error(f"Failed to generate cosmology plot: {e}")
        raise RuntimeError(f"Cosmology plot failed: {e}") from e


# -----------------------------------------------------------------------
# Rotation curves
# -----------------------------------------------------------------------

def plot_rotation_curve(
    r_obs: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_pred: np.ndarray,
    v_components: Optional[Dict[str, np.ndarray]] = None,
    galaxy_name: str = "Galaxy",
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    """Plot observed vs predicted rotation curve with components.

    Args:
        r_obs: Radii [kpc].
        v_obs: Observed velocities [km/s].
        v_err: Velocity uncertainties [km/s].
        v_pred: Total predicted velocity [km/s].
        v_components: Optional dict of named velocity components
            (e.g. {'disk': ..., 'gas': ..., 'DM halo': ...}).
        galaxy_name: Name for the plot title.

    Returns:
        Path to the saved image.
    """
    _ensure_dir(output_dir)
    filename = os.path.join(output_dir, f"rotation_curve_{galaxy_name.replace(' ', '_')}.png")
    logger.info(f"Generating rotation curve plot at {filename}...")

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.errorbar(
            r_obs, v_obs, yerr=v_err,
            fmt="o", color="black", markersize=4, capsize=2,
            label="Observed", zorder=5,
        )
        ax.plot(r_obs, v_pred, color="tab:red", linewidth=2.5, label="Total model", zorder=4)

        if v_components:
            colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple"]
            for (name, v_comp), color in zip(v_components.items(), colors):
                ax.plot(r_obs, v_comp, color=color, linestyle="--", linewidth=1.5, label=name)

        ax.set_xlabel("Radius [kpc]")
        ax.set_ylabel("Rotation velocity [km/s]")
        ax.set_title(f"Rotation Curve — {galaxy_name}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        fig.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        logger.info("Rotation curve plot generated successfully.")
        return filename

    except Exception as e:
        logger.error(f"Failed to generate rotation curve plot: {e}")
        raise RuntimeError(f"Rotation curve plot failed: {e}") from e


# -----------------------------------------------------------------------
# MCMC diagnostics
# -----------------------------------------------------------------------

def plot_mcmc_trace(
    chain: np.ndarray,
    param_names: Optional[list] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    """Plot MCMC trace (walker trajectories) for each parameter.

    Args:
        chain: Shape (n_steps, n_walkers, n_dim).
        param_names: List of parameter names.

    Returns:
        Path to the saved image.
    """
    _ensure_dir(output_dir)
    filename = os.path.join(output_dir, "mcmc_trace.png")

    n_steps, n_walkers, n_dim = chain.shape
    if param_names is None:
        param_names = [f"p{i}" for i in range(n_dim)]

    try:
        fig, axes = plt.subplots(n_dim, 1, figsize=(10, 3 * n_dim), sharex=True)
        if n_dim == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            for w in range(min(n_walkers, 20)):  # plot at most 20 walkers
                ax.plot(chain[:, w, i], alpha=0.3, linewidth=0.5)
            ax.set_ylabel(param_names[i])
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Step")
        fig.suptitle("MCMC Trace Plot", fontsize=14)
        fig.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()
        return filename

    except Exception as e:
        logger.error(f"Failed to generate trace plot: {e}")
        raise RuntimeError(f"Trace plot failed: {e}") from e


def plot_mcmc_corner(
    samples: np.ndarray,
    param_names: Optional[list] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    """Simple corner-style plot (1D histograms + 2D scatter) for MCMC posteriors.

    This is a lightweight alternative to the `corner` package.

    Args:
        samples: Flat samples, shape (n_samples, n_dim).
        param_names: List of parameter names.

    Returns:
        Path to the saved image.
    """
    _ensure_dir(output_dir)
    filename = os.path.join(output_dir, "mcmc_corner.png")

    n_dim = samples.shape[1]
    if param_names is None:
        param_names = [f"p{i}" for i in range(n_dim)]

    try:
        fig, axes = plt.subplots(n_dim, n_dim, figsize=(3 * n_dim, 3 * n_dim))
        if n_dim == 1:
            axes = np.array([[axes]])

        for i in range(n_dim):
            for j in range(n_dim):
                ax = axes[i, j]
                if j > i:
                    ax.set_visible(False)
                    continue
                if i == j:
                    ax.hist(samples[:, i], bins=40, color="tab:blue", alpha=0.7, density=True)
                else:
                    ax.scatter(
                        samples[:, j], samples[:, i],
                        s=1, alpha=0.1, color="tab:blue", rasterized=True,
                    )

                if i == n_dim - 1:
                    ax.set_xlabel(param_names[j])
                if j == 0:
                    ax.set_ylabel(param_names[i])

        fig.suptitle("Parameter Posteriors", fontsize=14)
        fig.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()
        return filename

    except Exception as e:
        logger.error(f"Failed to generate corner plot: {e}")
        raise RuntimeError(f"Corner plot failed: {e}") from e
