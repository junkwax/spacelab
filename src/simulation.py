import yaml
import logging
from src.models.dark_matter import DarkMatter
from src.models.dark_energy import QuintessenceField
from src.models.spacetime import SpacetimeGeometry
#from src.utils.data_loader import load_h5_data
#from src.utils.plotting import plot_rotation_curve
import numpy as np
import scipy.integrate as integrate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def einstein_field_equations(r, T_munu, spacetime):
    """Solves the simplified Einstein field equations for a static,
    spherically symmetric spacetime.

    Args:
        r (np.ndarray): Radial coordinate values.
        T_munu (np.ndarray): Total stress-energy tensor.
        spacetime (SpacetimeGeometry): The spacetime object.

    Returns:
        tuple: (A(r), B(r)) - the metric functions.
    """
    def equations(y, r, T_tt, T_rr, spacetime):
        A, B, dA_dr, dB_dr = y

        # Ensure A and B are not zero
        A = np.maximum(A, 1e-6)
        B = np.maximum(B, 1e-6)
        mass_kg = spacetime.mass * spacetime.solar_mass_kg
        rs = 2 * spacetime.G * mass_kg / (spacetime.c ** 2)

        d2A_dr2 = (8 * np.pi * spacetime.G * T_tt / (spacetime.c**4)  + (1-A)/(r**2) + dA_dr/(r*A))*A**2
        d2B_dr2 = B * ( dB_dr**2 / (2*B**2) + dB_dr/(r*B) + dA_dr*dB_dr/(2*A*B) - dA_dr/(r*A) + 8 * np.pi * spacetime.G * T_rr  / (spacetime.c**4) * A)

        return [dA_dr, dB_dr, d2A_dr2, d2B_dr2]

    # Extract T_tt and T_rr from the total stress-energy tensor
    T_tt = T_munu[0, 0, :]
    T_rr = T_munu[1, 1, :]

    # Initial conditions for A(r) and B(r) (Schwarzschild)
    mass_kg = spacetime.mass * spacetime.solar_mass_kg
    rs = 2 * spacetime.G * mass_kg / (spacetime.c ** 2)
    y0 = [1/(1-rs/r[0]), -(1-rs/r[0]), (rs/(r[0]**2))/(1-rs/r[0]), -rs/(r[0]**2)]


    # Solve the equations
    solution = integrate.odeint(equations, y0, r, args=(T_tt, T_rr, spacetime), rtol=1e-6, atol=1e-8)

    # Extract A(r) and B(r)
    A = solution[:, 0]
    B = solution[:, 1]
    return A, B

def run_simulation(config_file):
    """Run a black hole simulation with dark matter and dark energy.

    Args:
        config_file (str): Path to the YAML configuration file.

    Raises:
        ValueError: If the configuration is invalid.
    """
    logger.info("Starting simulation...")
    try:
        # Load configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Initialize models
        dm = DarkMatter(
            mass=config['dark_matter']['mass'],
            coupling_dilaton=config['dark_matter']['coupling_dilaton'],
            coupling_curvature=config['dark_matter']['coupling_curvature']
        )
        de = QuintessenceField(
            V0=config['dark_energy']['V0'],
            lambda_=config['dark_energy']['lambda_']
        )
        spacetime = SpacetimeGeometry(mass=config['spacetime']['black_hole_mass'])

        # Set initial conditions (TODO: Determine appropriate values)
        initial_phi_DM = 0.1
        initial_dphi_DM_dr = 0.0
        initial_phi_DE = 0.0
        initial_dphi_DE_dt = 0.0
        initial_dphi_DE_dr = 0.0
        initial_dilaton = 0.0 # Initial dilaton value
        initial_graviphoton = 0.0

        # Define radial coordinate range
        r_range = np.linspace(2.1, 10.0, 100)
        t_range = np.linspace(0, 10, 100) # Add time range

        # --- Numerical Integration ---
        def coupled_field_equations(y, r, dilaton_field, graviphoton_field):
            phi_DM, dphi_DM_dr, phi_DE, dphi_DE_dr, dphi_DE_dt = y

            # Calculate dark matter field equation
            dphi_DM_dr, ddphi_DM_dr2 = dm.field_equation((phi_DM, dphi_DM_dr), r, dilaton_field, graviphoton_field, phi_DE)

            # Calculate dark energy field equation
            dphi_DE_dr, ddphi_DE_dr2, dphi_DE_dt, ddphi_DE_dt2 = de.field_equation((phi_DE, dphi_DE_dr, dphi_DE_dt), r, dilaton_field, phi_DM, config['dark_matter']['coupling_dilaton'])


            return [dphi_DM_dr, ddphi_DM_dr2, dphi_DE_dr, ddphi_DE_dr2, ddphi_DE_dt2]



        # --- Iterative Solution ---
        max_iterations = 10  # Set a maximum number of iterations
        tolerance = 1e-4  # Set a convergence tolerance

        # Initial guess for the metric (Schwarzschild)
        g_tt, g_rr, _, _ = spacetime.schwarzschild_metric(r_range)
        A_current = 1/g_rr #Using previous definition
        B_current = -g_tt
        dilaton_field = initial_dilaton * np.ones_like(r_range) # Start with zero
        graviphoton_field = initial_graviphoton * np.ones_like(r_range)

        for iteration in range(max_iterations):
            logger.info(f"Iteration: {iteration + 1}")
            # Solve the coupled field equations
            initial_conditions = [initial_phi_DM, initial_dphi_DM_dr, initial_phi_DE, initial_dphi_DE_dr, initial_dphi_DE_dt]
            solution = integrate.odeint(coupled_field_equations, initial_conditions, r_range, args=(dilaton_field, graviphoton_field))

            # Extract solutions
            phi_DM_solution = solution[:, 0]
            dphi_DM_dr_solution = solution[:, 1]
            phi_DE_solution = solution[:, 2]
            dphi_DE_dr_solution = solution[:, 3]
            dphi_DE_dt_solution = solution[:, 4]

            # Calculate the total stress-energy tensor
            T_munu_DM = spacetime.stress_energy_tensor_DM(r_range, phi_DM_solution, dphi_DM_dr_solution,
                                                        phi_DE_solution, dilaton_field,
                                                        config['dark_matter']['coupling_dilaton'])
            T_munu_DE = spacetime.stress_energy_tensor_DE(r_range, phi_DE_solution, dphi_DE_dr_solution,
                                                        dphi_DE_dt_solution, phi_DM_solution,
                                                        dilaton_field, config['dark_matter']['coupling_dilaton'],
                                                        config['dark_energy']['V0'], config['dark_energy']['lambda_'])
            T_munu_total = T_munu_DM + T_munu_DE


            # Solve Einstein Field Equations
            A_new, B_new = einstein_field_equations(r_range, T_munu_total, spacetime)

            #Check for convergence
            A_diff = np.max(np.abs((A_new - A_current)/A_current))
            B_diff = np.max(np.abs((B_new - B_current)/B_current))
            if A_diff < tolerance and B_diff < tolerance:
                logger.info("Convergence achieved.")
                break

            # Update metric for next iteration
            A_current = A_new
            B_current = B_new

            # Update the spacetime object with new metric.  This is VERY important
            spacetime.schwarzschild_metric = lambda r: ( -B_current, A_current, r**2, (r**2)*(np.sin(np.pi/2)**2) )

        else:
            logger.warning("Maximum number of iterations reached without convergence.")

        # --- Plotting ---
        # Placeholder for plotting (TODO: Implement)
        #...

        logger.info("Simulation completed successfully.")

    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    run_simulation(args.config)
