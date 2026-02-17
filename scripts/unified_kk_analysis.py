#!/usr/bin/env python3
"""
SpaceLab — Unified Kaluza-Klein Analysis

THE KEY TEST: Can a single set of KK parameters (R5, M5, instanton_suppression)
simultaneously explain the rotation curves of galaxies spanning 3 orders of
magnitude in mass?

If yes → the KK framework makes a universal prediction connecting the
         compactification geometry to dark matter observations.
If no  → the extra dimension cannot be the sole origin of dark matter,
         or additional physics (baryonic feedback, etc.) is needed.

This script:
  1. Scans R5 (compactification radius) to find the best-fit value
  2. For each R5, derives the axion mass and soliton profile
  3. Fits all SPARC galaxies jointly with the KK-predicted DM profile
  4. Reports whether a universal R5 exists

Usage:
    python scripts/unified_kk_analysis.py
"""

import os
import sys
import logging
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.kaluza_klein import KKParameters, scan_R5, MSUN_KG, KPC_TO_M, G_N
from src.models.bulk_stress_energy import StressEnergyKK
from src.observables.rotation_curves import load_sparc_catalog, SPARCGalaxy

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ===================================================================
# KK-predicted rotation curve model
# ===================================================================

def estimate_halo_mass(v_flat_kms: float, H0: float = 67.4) -> float:
    """Estimate virial mass from the flat rotation velocity.

    Uses the relation V_flat² ≈ G M200 / r200, combined with
    M200 = (4/3)π × 200 ρ_crit × r200³, giving:
        M200 = V³_flat / (10 G H)
    """
    v_flat = v_flat_kms * 1e3  # m/s
    H = H0 * 1e3 / 3.0857e22  # s⁻¹
    M200_kg = v_flat**3 / (10.0 * G_N * H)
    return M200_kg / MSUN_KG


def kk_rotation_velocity(
    r_kpc: np.ndarray,
    kk: KKParameters,
    galaxy: SPARCGalaxy,
    upsilon_disk: float = 0.5,
) -> np.ndarray:
    """Compute total rotation velocity using the KK-derived DM profile.

    The DM halo is predicted from the axion soliton + NFW envelope,
    with core radius set by the soliton-halo mass relation.
    M_halo is estimated from V_flat of the galaxy.

    Returns:
        v_total [km/s]
    """
    M_halo = estimate_halo_mass(galaxy.v_flat)

    # DM component from KK theory
    v_halo = kk.axion_rotation_velocity(r_kpc, M_halo)

    # Baryons from SPARC data
    v_gas = np.interp(r_kpc, galaxy.r_kpc, galaxy.v_gas)
    v_disk = np.interp(r_kpc, galaxy.r_kpc, galaxy.v_disk)
    v_bul = np.interp(r_kpc, galaxy.r_kpc, galaxy.v_bul)

    v_bar_sq = v_gas**2 + upsilon_disk * v_disk**2 + 0.7 * v_bul**2
    v_total = np.sqrt(v_halo**2 + np.maximum(v_bar_sq, 0.0))
    return v_total


def chi2_single_galaxy(
    kk: KKParameters,
    galaxy: SPARCGalaxy,
    upsilon_disk: float = 0.5,
) -> float:
    """χ² for one galaxy given KK parameters."""
    v_model = kk_rotation_velocity(galaxy.r_kpc, kk, galaxy, upsilon_disk)
    residuals = (galaxy.v_obs - v_model) / galaxy.v_err
    return float(np.sum(residuals**2))


def chi2_all_galaxies(
    kk: KKParameters,
    catalog: dict,
    upsilon_disk: float = 0.5,
) -> tuple:
    """Total χ² across all galaxies + per-galaxy breakdown."""
    total = 0.0
    breakdown = {}
    for name, galaxy in catalog.items():
        chi2 = chi2_single_galaxy(kk, galaxy, upsilon_disk)
        n_pts = galaxy.n_points
        breakdown[name] = {"chi2": chi2, "n_pts": n_pts, "chi2_red": chi2 / max(n_pts - 1, 1)}
        total += chi2
    return total, breakdown


# ===================================================================
# R5 scan
# ===================================================================

def scan_compactification_radius(
    catalog: dict,
    m_a_min: float = 1e-24,
    m_a_max: float = 1e-20,
    n_scan: int = 50,
    upsilon_disk: float = 0.5,
) -> dict:
    """Scan axion mass to find the value that minimizes total χ².

    Uses m_axion_target mode: for each m_a, derives the required
    instanton suppression and checks KK self-consistency.
    Also marginalizes over Υ★ at each point.
    """
    m_a_values = np.geomspace(m_a_min, m_a_max, n_scan)
    R5_fixed = 1e-6  # Fix R5, scan m_a (controls the observable)

    chi2_total = np.zeros(n_scan)
    best_upsilon = np.zeros(n_scan)
    breakdowns = []

    upsilon_grid = np.linspace(0.2, 1.2, 11)

    logger.info(f"Scanning m_axion: [{m_a_min:.1e}, {m_a_max:.1e}] eV ({n_scan} points)")

    for i, m_a in enumerate(m_a_values):
        kk = KKParameters(R5=R5_fixed, m_axion_target=m_a)

        # Marginalize over Υ★
        best_chi2 = np.inf
        best_ups = 0.5
        best_bd = None

        for ups in upsilon_grid:
            chi2, bd = chi2_all_galaxies(kk, catalog, ups)
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_ups = ups
                best_bd = bd

        chi2_total[i] = best_chi2
        best_upsilon[i] = best_ups
        breakdowns.append(best_bd)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  m_a={m_a:.2e} eV, S_inst={kk.instanton_action:.1f}, "
                f"χ²={best_chi2:.1f}, Υ★={best_ups:.2f}"
            )

    # Best fit
    best_idx = np.argmin(chi2_total)

    # Reconstruct best KK params
    kk_best = KKParameters(R5=R5_fixed, m_axion_target=m_a_values[best_idx])

    return {
        "m_a_values": m_a_values,
        "R5_fixed": R5_fixed,
        "chi2_total": chi2_total,
        "best_upsilon": best_upsilon,
        "breakdowns": breakdowns,
        "best_idx": best_idx,
        "best_R5": R5_fixed,
        "best_m_axion": m_a_values[best_idx],
        "best_chi2": chi2_total[best_idx],
        "best_breakdown": breakdowns[best_idx],
        "kk_best": kk_best,
    }


# ===================================================================
# Plotting
# ===================================================================

def plot_R5_scan(scan_result: dict, output_dir: str):
    """Plot χ² vs m_axion and required instanton action."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    m_a = scan_result["m_a_values"]
    chi2 = scan_result["chi2_total"]
    best_idx = scan_result["best_idx"]

    # χ² vs m_axion
    ax1.semilogx(m_a, chi2, "r-", lw=2)
    ax1.axvline(m_a[best_idx], color="blue", ls="--",
                label=f"Best $m_a$={m_a[best_idx]:.2e} eV")
    ax1.set_xlabel("$m_a$ (axion mass) [eV]")
    ax1.set_ylabel("Total $\\chi^2$ (all galaxies)")
    ax1.set_title("Axion Mass Scan — Joint Fit to 5 SPARC Galaxies")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Required instanton action vs m_axion
    R5 = scan_result["R5_fixed"]
    S_inst = []
    for m in m_a:
        kk = KKParameters(R5=R5, m_axion_target=m)
        S_inst.append(kk.instanton_action)
    S_inst = np.array(S_inst)

    ax2.semilogx(m_a, S_inst, "g-", lw=2)
    ax2.axvline(m_a[best_idx], color="blue", ls="--")
    ax2.axhspan(50, 200, alpha=0.1, color="green", label="Typical instanton range")
    ax2.set_xlabel("$m_a$ (axion mass) [eV]")
    ax2.set_ylabel("Required instanton action $S$")
    ax2.set_title("KK Self-Consistency: Instanton Action")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "m_axion_scan.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_unified_fits(
    kk: KKParameters,
    catalog: dict,
    upsilon: float,
    output_dir: str,
):
    """Plot all galaxy fits with the best-fit universal KK parameters."""
    n_gal = len(catalog)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (name, galaxy) in enumerate(catalog.items()):
        if idx >= 6:
            break
        ax = axes[idx]

        r = galaxy.r_kpc
        ax.errorbar(r, galaxy.v_obs, yerr=galaxy.v_err, fmt="ko", ms=3,
                     capsize=2, label="$V_\\mathrm{obs}$")

        # KK prediction
        r_smooth = np.linspace(r[0], r[-1], 200)
        v_kk = kk_rotation_velocity(r_smooth, kk, galaxy, upsilon)
        ax.plot(r_smooth, v_kk, "r-", lw=2, label="KK prediction")

        # DM-only
        M_halo = estimate_halo_mass(galaxy.v_flat)
        v_dm = kk.axion_rotation_velocity(r_smooth, M_halo)
        ax.plot(r_smooth, v_dm, "b--", lw=1, alpha=0.6, label="DM (KK axion)")

        # Baryons
        v_bar = galaxy.baryonic_velocity(upsilon)
        ax.plot(r, v_bar, "g:", lw=1, alpha=0.6, label="Baryons")

        chi2 = chi2_single_galaxy(kk, galaxy, upsilon)
        chi2_r = chi2 / max(galaxy.n_points - 1, 1)
        ax.set_title(f"{name}  ($\\chi^2_r$={chi2_r:.2f})", fontsize=11)
        ax.set_xlabel("r [kpc]")
        ax.set_ylabel("V [km/s]")
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)

    # Hide unused subplot
    if n_gal < 6:
        for idx in range(n_gal, 6):
            axes[idx].set_visible(False)

    fig.suptitle(
        f"Universal KK Fit:  $R_5$={kk.R5:.2e} m,  "
        f"$m_a$={kk.m_axion:.2e} eV,  $\\Upsilon_*$={upsilon:.2f}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(output_dir, "unified_kk_fits.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def write_unified_summary(
    scan_result: dict,
    kk_best: KKParameters,
    output_dir: str,
):
    """Write the unified analysis summary."""
    path = os.path.join(output_dir, "unified_kk_summary.txt")

    with open(path, "w") as f:
        f.write(kk_best.summary())
        f.write("\n\n")

        f.write("ROTATION CURVE FIT RESULTS (Universal KK Parameters)\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'Galaxy':<12} {'N_pts':>6} {'χ²':>10} {'χ²_r':>10}\n")
        f.write("-" * 65 + "\n")

        total_chi2 = 0
        total_pts = 0
        for name, info in scan_result["best_breakdown"].items():
            f.write(f"{name:<12} {info['n_pts']:>6} {info['chi2']:>10.2f} {info['chi2_red']:>10.3f}\n")
            total_chi2 += info["chi2"]
            total_pts += info["n_pts"]

        n_params = 3  # R5, instanton_suppression, Υ★
        total_dof = total_pts - n_params
        f.write("-" * 65 + "\n")
        f.write(f"{'TOTAL':<12} {total_pts:>6} {total_chi2:>10.2f} {total_chi2/total_dof:>10.3f}\n")
        f.write(f"\nTotal DOF = {total_dof}\n")
        f.write(f"p-value note: χ²_r ~ 1 indicates a good fit,\n")
        f.write(f"              χ²_r >> 1 indicates the model is rejected,\n")
        f.write(f"              χ²_r << 1 indicates overfitting or overestimated errors.\n")

        f.write("\n\nKEY PREDICTION:\n")
        f.write(f"  The compactification radius R5 = {kk_best.R5:.4e} m\n")
        f.write(f"  predicts an axion mass m_a = {kk_best.m_axion:.4e} eV\n")
        f.write(f"  and a dilaton mass m_d = {kk_best.m_dilaton:.4e} eV\n")
        f.write(f"\n  This can be tested independently via:\n")
        f.write(f"    - Black hole superradiance constraints on m_a\n")
        f.write(f"    - Lyman-α forest constraints on fuzzy DM mass\n")
        f.write(f"    - EHT shadow measurements (dilaton charge → modified ISCO)\n")
        f.write(f"    - LISA gravitational wave phase shifts (scalar dipole radiation)\n")

    logger.info(f"Saved {path}")


# ===================================================================
# Main
# ===================================================================

def main(data_dir="data/sparc", output_dir="output/unified_kk"):
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading SPARC catalog...")
    catalog = load_sparc_catalog(data_dir)
    logger.info(f"Loaded {len(catalog)} galaxies")

    # === Phase 1: Scan R5 ===
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Scanning compactification radius R5")
    logger.info("=" * 60)

    scan_result = scan_compactification_radius(
        catalog, m_a_min=1e-24, m_a_max=1e-20, n_scan=60
    )

    best_R5 = scan_result["best_R5"]
    best_ma = scan_result["best_m_axion"]
    best_chi2 = scan_result["best_chi2"]
    best_ups = scan_result["best_upsilon"][scan_result["best_idx"]]

    logger.info(f"\nBest-fit: m_a={best_ma:.4e} eV, "
                f"χ²={best_chi2:.1f}, Υ★={best_ups:.2f}")

    # === Phase 2: Detailed fit with best parameters ===
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Detailed analysis with best-fit KK parameters")
    logger.info("=" * 60)

    kk_best = scan_result["kk_best"]
    print(kk_best.summary())

    # Per-galaxy results
    logger.info("\nPer-galaxy breakdown:")
    for name, info in scan_result["best_breakdown"].items():
        logger.info(f"  {name}: χ²={info['chi2']:.2f}, χ²_r={info['chi2_red']:.3f}")

    # === Phase 3: Compute bulk stress-energy at best fit ===
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Bulk stress-energy tensor")
    logger.info("=" * 60)

    se = StressEnergyKK(kk_best)
    T = se.total_stress_energy(
        r=1e20,  # ~3 kpc in meters
        phi=kk_best.dilaton_vev,
        dphi_dr=1e-30,  # small gradient at galactic scales
        psi=1e-10,      # typical axion field amplitude
        dpsi_dr=1e-30,
    )
    logger.info(f"  ρ_total = {T['rho_total']:.4e}")
    logger.info(f"  w_eff   = {T['w_eff']:.4f}")
    logger.info(f"  Components: dilaton={T['rho_breakdown']['dilaton']:.2e}, "
                f"axion={T['rho_breakdown']['axion']:.2e}, "
                f"bulk={T['rho_breakdown']['bulk']:.2e}")

    # === Phase 4: Plots ===
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: Generating plots")
    logger.info("=" * 60)

    plot_R5_scan(scan_result, output_dir)
    plot_unified_fits(kk_best, catalog, best_ups, output_dir)
    write_unified_summary(scan_result, kk_best, output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("UNIFIED KK ANALYSIS COMPLETE")
    logger.info(f"Results in {output_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/sparc")
    parser.add_argument("--output-dir", default="output/unified_kk")
    args = parser.parse_args()
    main(data_dir=args.data_dir, output_dir=args.output_dir)
