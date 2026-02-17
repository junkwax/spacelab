#!/usr/bin/env python3
"""
SpaceLab — SPARC Rotation Curve Analysis

Loads real SPARC galaxy data, fits both NFW and axion-soliton dark matter
models via MCMC, and produces comparison plots and summary statistics.

Usage:
    python scripts/analyze_sparc.py [--data-dir data/sparc] [--steps 5000]

Output:
    output/sparc_results/
        <galaxy>_rotation_fit.png   — per-galaxy rotation curve plots
        <galaxy>_residuals.png      — per-galaxy residual plots
        model_comparison.png        — NFW vs axion χ² comparison
        summary.txt                 — parameter table for all galaxies
"""

import os
import sys
import logging
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.observables.rotation_curves import (
    load_sparc_catalog,
    build_nfw_model,
    build_axion_model,
    SPARCGalaxy,
)
from src.fitting.likelihood import log_posterior, reduced_chi2
from src.fitting.mcmc import MCMCSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Fitting engine
# ======================================================================

def fit_galaxy(
    galaxy: SPARCGalaxy,
    model_type: str = "nfw",
    n_steps: int = 5000,
    burn_in: int = 1000,
    seed: int = 42,
) -> dict:
    """Fit a DM halo model to a single SPARC galaxy.

    Args:
        galaxy: SPARCGalaxy instance.
        model_type: 'nfw' or 'axion'.
        n_steps: MCMC steps.
        burn_in: Burn-in steps to discard.
        seed: Random seed.

    Returns:
        Dict with 'sampler', 'model_fn', 'param_names', 'priors',
        'best_fit', 'chi2_red'.
    """
    if model_type == "nfw":
        model_fn, param_names, priors = build_nfw_model(galaxy)
    elif model_type == "axion":
        model_fn, param_names, priors = build_axion_model(galaxy)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    data = galaxy.as_dict()

    def log_prob(params):
        return log_posterior(params, data, model_fn, param_names, priors)

    sampler = MCMCSampler(
        log_prob_fn=log_prob,
        param_names=param_names,
        priors=priors,
        backend="metropolis",
    )

    chain = sampler.run(n_steps=n_steps, burn_in=burn_in, seed=seed)
    best = sampler.get_best_fit()

    chi2_r = reduced_chi2(best, data, model_fn, param_names)

    return {
        "sampler": sampler,
        "model_fn": model_fn,
        "param_names": param_names,
        "priors": priors,
        "best_fit": best,
        "chi2_red": chi2_r,
    }


# ======================================================================
# Plotting
# ======================================================================

def plot_rotation_fit(
    galaxy: SPARCGalaxy,
    nfw_result: dict,
    axion_result: dict,
    output_dir: str,
):
    """Plot the rotation curve with NFW and axion model fits."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={"hspace": 0.05})

    r = galaxy.r_kpc

    # --- Data ---
    ax1.errorbar(r, galaxy.v_obs, yerr=galaxy.v_err, fmt="ko", ms=4,
                 capsize=2, label="$V_\\mathrm{obs}$", zorder=5)

    # --- Baryonic components ---
    v_bar = galaxy.baryonic_velocity(upsilon_disk=0.5)
    ax1.plot(r, galaxy.v_gas, ":", color="green", lw=1, alpha=0.7, label="$V_\\mathrm{gas}$")
    ax1.plot(r, np.sqrt(0.5) * galaxy.v_disk, "--", color="orange", lw=1, alpha=0.7,
             label="$V_\\mathrm{disk}$ ($\\Upsilon_*$=0.5)")
    if np.any(galaxy.v_bul > 0):
        ax1.plot(r, np.sqrt(0.7) * galaxy.v_bul, "-.", color="brown", lw=1, alpha=0.7,
                 label="$V_\\mathrm{bul}$ ($\\Upsilon_*$=0.7)")

    # --- NFW fit ---
    r_smooth = np.linspace(r[0], r[-1], 200)
    best_nfw = dict(zip(nfw_result["param_names"], nfw_result["best_fit"]))
    v_nfw = nfw_result["model_fn"](r_smooth, **best_nfw)
    ax1.plot(r_smooth, v_nfw, "-", color="tab:blue", lw=2,
             label=f"NFW ($\\chi^2_r$={nfw_result['chi2_red']:.2f})")

    # --- Axion fit ---
    best_ax = dict(zip(axion_result["param_names"], axion_result["best_fit"]))
    v_ax = axion_result["model_fn"](r_smooth, **best_ax)
    ax1.plot(r_smooth, v_ax, "-", color="tab:red", lw=2,
             label=f"Axion ($\\chi^2_r$={axion_result['chi2_red']:.2f})")

    ax1.set_ylabel("$V_\\mathrm{rot}$ [km/s]")
    ax1.set_title(f"{galaxy.name} — Rotation Curve Decomposition", fontsize=13)
    ax1.legend(loc="lower right", fontsize=8, ncol=2)
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=0.3)

    # --- Residuals ---
    v_nfw_data = nfw_result["model_fn"](r, **best_nfw)
    v_ax_data = axion_result["model_fn"](r, **best_ax)
    res_nfw = (galaxy.v_obs - v_nfw_data) / galaxy.v_err
    res_ax = (galaxy.v_obs - v_ax_data) / galaxy.v_err

    ax2.axhline(0, color="gray", lw=0.8)
    ax2.errorbar(r - 0.05, res_nfw, yerr=1.0, fmt="s", color="tab:blue", ms=3,
                 capsize=1, label="NFW", alpha=0.8)
    ax2.errorbar(r + 0.05, res_ax, yerr=1.0, fmt="^", color="tab:red", ms=3,
                 capsize=1, label="Axion", alpha=0.8)
    ax2.set_xlabel("Radius [kpc]")
    ax2.set_ylabel("Residual [$\\sigma$]")
    ax2.legend(fontsize=8)
    ax2.set_ylim(-4, 4)
    ax2.grid(alpha=0.3)

    filepath = os.path.join(output_dir, f"{galaxy.name}_rotation_fit.png")
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {filepath}")


def plot_model_comparison(all_results: dict, output_dir: str):
    """Bar chart comparing NFW vs axion χ²_red across galaxies."""
    names = list(all_results.keys())
    chi2_nfw = [all_results[n]["nfw"]["chi2_red"] for n in names]
    chi2_ax = [all_results[n]["axion"]["chi2_red"] for n in names]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w / 2, chi2_nfw, w, label="NFW", color="tab:blue", alpha=0.8)
    bars2 = ax.bar(x + w / 2, chi2_ax, w, label="Axion", color="tab:red", alpha=0.8)

    ax.axhline(1.0, color="gray", ls="--", lw=1, label="$\\chi^2_r = 1$")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Reduced $\\chi^2$")
    ax.set_title("Model Comparison: NFW vs Axion Soliton")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    filepath = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {filepath}")


# ======================================================================
# Summary
# ======================================================================

def write_summary(all_results: dict, output_dir: str):
    """Write a text summary of all fits."""
    filepath = os.path.join(output_dir, "summary.txt")

    with open(filepath, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("SpaceLab — SPARC Rotation Curve Fit Summary\n")
        f.write("=" * 90 + "\n\n")

        for name, res in all_results.items():
            f.write(f"--- {name} ---\n")
            for model_type in ["nfw", "axion"]:
                r = res[model_type]
                f.write(f"  {model_type.upper()} (χ²_r = {r['chi2_red']:.3f}):\n")
                for pname, val in zip(r["param_names"], r["best_fit"]):
                    f.write(f"    {pname:>15s} = {val:.4e}\n")
            f.write("\n")

        # Overall comparison
        f.write("=" * 90 + "\n")
        f.write("Model Comparison Summary\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Galaxy':<15} {'NFW χ²_r':>10} {'Axion χ²_r':>12} {'Better':>10}\n")
        f.write("-" * 90 + "\n")
        for name, res in all_results.items():
            nfw_chi2 = res["nfw"]["chi2_red"]
            ax_chi2 = res["axion"]["chi2_red"]
            better = "NFW" if nfw_chi2 < ax_chi2 else "Axion"
            f.write(f"{name:<15} {nfw_chi2:>10.3f} {ax_chi2:>12.3f} {better:>10}\n")

    logger.info(f"Saved {filepath}")


# ======================================================================
# Main
# ======================================================================

def main(data_dir: str = "data/sparc", n_steps: int = 5000, output_dir: str = "output/sparc_results"):
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading SPARC catalog from {data_dir}...")
    catalog = load_sparc_catalog(data_dir)
    logger.info(f"Found {len(catalog)} galaxies: {list(catalog.keys())}")

    all_results = {}

    for name, galaxy in catalog.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Fitting {name} ({galaxy.n_points} points, Vflat≈{galaxy.v_flat:.0f} km/s)")
        logger.info(f"{'='*60}")

        # Fit NFW
        logger.info(f"  Fitting NFW model...")
        nfw_res = fit_galaxy(galaxy, model_type="nfw", n_steps=n_steps)
        logger.info(f"  NFW: χ²_r = {nfw_res['chi2_red']:.3f}")
        nfw_res["sampler"].summary()

        # Fit axion
        logger.info(f"  Fitting Axion model...")
        axion_res = fit_galaxy(galaxy, model_type="axion", n_steps=n_steps, seed=123)
        logger.info(f"  Axion: χ²_r = {axion_res['chi2_red']:.3f}")
        axion_res["sampler"].summary()

        all_results[name] = {"nfw": nfw_res, "axion": axion_res}

        # Plot
        plot_rotation_fit(galaxy, nfw_res, axion_res, output_dir)

    # Summary
    plot_model_comparison(all_results, output_dir)
    write_summary(all_results, output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SpaceLab SPARC rotation curve analysis")
    parser.add_argument("--data-dir", default="data/sparc")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--output-dir", default="output/sparc_results")
    args = parser.parse_args()

    main(data_dir=args.data_dir, n_steps=args.steps, output_dir=args.output_dir)
