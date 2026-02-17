"""
SpaceLab simulation driver.

Provides three main workflows:
  1. run_bh_simulation     — Black hole scalar field profile (original)
  2. run_cosmology         — FRW + quintessence evolution
  3. run_rotation_fit      — Fit rotation curves via MCMC
"""

import yaml
import logging
import argparse
import numpy as np
import scipy.integrate as integrate
from typing import Dict, Optional

from src.models.dark_matter import DarkMatter
from src.models.dark_energy import QuintessenceField
from src.models.spacetime import SpacetimeGeometry
from src.models.cosmology import FRWCosmology
from src.observables.rotation_curves import (
    load_sparc_catalog,
    build_nfw_model,
    generate_synthetic_sparc,
    nfw_velocity,
    KPC_TO_M,
    MSUN_KG,
)
from src.fitting.likelihood import log_posterior
from src.fitting.mcmc import MCMCSampler
from src.utils.plotting import plot_scalar_field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ======================================================================
# 1. Black hole scalar field simulation (original, updated)
# ======================================================================

def run_bh_simulation(config_file: str) -> np.ndarray:
    """Run a black hole scalar field simulation."""
    logger.info("Starting BH simulation...")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    dm = DarkMatter(
        mass=float(config["dark_matter"]["mass"]),
        coupling_dilaton=float(config["dark_matter"]["coupling_dilaton"]),
        coupling_curvature=float(config["dark_matter"]["coupling_curvature"]),
    )
    de = QuintessenceField(
        V0=float(config["dark_energy"]["V0"]),
        lambda_=float(config["dark_energy"]["lambda_"]),
    )
    spacetime = SpacetimeGeometry(mass=float(config["spacetime"]["black_hole_mass"]))

    dilaton_field = float(config.get("fields", {}).get("dilaton", 1.0))
    graviphoton_field = float(config.get("fields", {}).get("graviphoton", 0.0))

    initial_phi = 0.1
    initial_dphi_dr = 0.0
    initial_phi_de = 0.0
    initial_dphi_de_dt = 0.0

    rs = spacetime._get_schwarzschild_radius()
    r_start = 1.5 * rs
    r_end = 50.0 * rs
    n_points = int(config.get("numerical", {}).get("grid_size", 100))
    r_range = np.linspace(r_start, r_end, n_points)

    logger.info(f"Rs: {rs:.3e} m  |  r: [{r_start:.3e}, {r_end:.3e}] m")

    def coupled_field_equations(y, r):
        phi, dphi_dr, phi_de, dphi_de_dt = y
        dphi_dr_out, ddphi_dr2 = dm.field_equation(
            (phi, dphi_dr), r, dilaton_field, graviphoton_field
        )
        # TODO: couple DE evolution once FRW background feeds in
        ddphi_de_dt2 = 0.0
        return [dphi_dr_out, ddphi_dr2, dphi_de_dt, ddphi_de_dt2]

    y0 = [initial_phi, initial_dphi_dr, initial_phi_de, initial_dphi_de_dt]
    solution = integrate.odeint(coupled_field_equations, y0, r_range)

    phi_solution = solution[:, 0]
    dphi_dr_solution = solution[:, 1]

    logger.info(f"BH simulation done. Final phi: {phi_solution[-1]:.6e}")
    plot_scalar_field(r_range, phi_solution, dphi_dr_solution)

    return solution


# ======================================================================
# 2. Cosmological evolution
# ======================================================================

def run_cosmology(config_file: str) -> Dict[str, np.ndarray]:
    """Evolve the FRW background with quintessence."""
    logger.info("Starting cosmological evolution...")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    de_cfg = config.get("dark_energy", {})
    cosmo_cfg = config.get("cosmology", {})

    quintessence = QuintessenceField(
        V0=float(de_cfg.get("V0", 1e-47)),
        lambda_=float(de_cfg.get("lambda_", 0.1)),
    )

    cosmo = FRWCosmology(
        H0=float(cosmo_cfg.get("H0", 67.4)),
        Omega_m0=float(cosmo_cfg.get("Omega_m0", 0.315)),
        Omega_r0=float(cosmo_cfg.get("Omega_r0", 9.1e-5)),
        quintessence=quintessence,
    )

    result = cosmo.evolve(
        a_start=float(cosmo_cfg.get("a_start", 0.001)),
        a_end=float(cosmo_cfg.get("a_end", 1.0)),
        phi_init=float(cosmo_cfg.get("phi_init", 0.0)),
        dphi_dt_init=float(cosmo_cfg.get("dphi_dt_init", 0.0)),
        n_points=int(cosmo_cfg.get("n_points", 1000)),
    )

    logger.info("Cosmological evolution complete.")
    return result


# ======================================================================
# 3. Rotation curve fitting
# ======================================================================

def run_rotation_fit(
    config_file: Optional[str] = None,
    data: Optional[Dict] = None,
    data_dir: str = "data/sparc",
    galaxy_name: Optional[str] = None,
    n_steps: int = 5000,
    burn_in: int = 1000,
) -> Dict:
    """Fit an NFW rotation curve model to data via MCMC.

    Can load real SPARC data or use a synthetic dataset.
    """
    logger.info("Starting rotation curve fitting...")

    if data is None and galaxy_name:
        # Load real SPARC data
        catalog = load_sparc_catalog(data_dir)
        if galaxy_name not in catalog:
            raise ValueError(f"Galaxy {galaxy_name} not found. Available: {list(catalog.keys())}")
        galaxy = catalog[galaxy_name]
        model_fn, param_names, priors = build_nfw_model(galaxy)
        data = galaxy.as_dict()
    elif data is None:
        # Synthetic fallback
        logger.info("No data provided — generating synthetic data")
        data = generate_synthetic_sparc(v_flat=150.0, n_points=30)
        param_names = ["log10_rho_s", "r_s_kpc"]
        priors = {"log10_rho_s": (5.0, 10.0), "r_s_kpc": (1.0, 100.0)}

        def model_fn(r_kpc, log10_rho_s, r_s_kpc):
            return nfw_velocity(r_kpc, 10**log10_rho_s, r_s_kpc)
    else:
        param_names = ["log10_rho_s", "r_s_kpc"]
        priors = {"log10_rho_s": (5.0, 10.0), "r_s_kpc": (1.0, 100.0)}

        def model_fn(r_kpc, log10_rho_s, r_s_kpc):
            return nfw_velocity(r_kpc, 10**log10_rho_s, r_s_kpc)

    def log_prob(params):
        return log_posterior(params, data, model_fn, param_names, priors)

    sampler = MCMCSampler(
        log_prob_fn=log_prob,
        param_names=param_names,
        priors=priors,
        backend="metropolis",
    )

    chain = sampler.run(n_steps=n_steps, burn_in=burn_in)
    results = sampler.summary()

    return {
        "sampler": sampler,
        "results": results,
        "data": data,
        "model_fn": model_fn,
    }


# Backward-compatible alias
run_simulation = run_bh_simulation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpaceLab simulation suite")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--mode",
        choices=["bh", "cosmology", "rotation"],
        default="bh",
    )
    parser.add_argument("--mcmc-steps", type=int, default=5000)
    args = parser.parse_args()

    if args.mode == "bh":
        run_bh_simulation(args.config)
    elif args.mode == "cosmology":
        run_cosmology(args.config)
    elif args.mode == "rotation":
        run_rotation_fit(args.config, n_steps=args.mcmc_steps)
