import sympy

# Define coordinates
t, r, theta, phi = sympy.symbols('t r theta phi')

# Define functions of r (dilaton and graviphoton)
phi_dilaton = sympy.Function('phi_dilaton')(r)
A_t = sympy.Function('A_t')(r)  # Assuming only A_t is non-zero

# 4D Schwarzschild metric components (symbolic)
M = sympy.symbols('M')
G = sympy.symbols('G')
c = sympy.symbols('c')
rs = 2 * G * M / c**2
gtt_4D = -(1 - rs/r)
grr_4D = 1 / (1 - rs/r)
gtheta_theta_4D = r**2
gphi_phi_4D = r**2 * sympy.sin(theta)**2

# 4D Inverse Schwarzschild metric
g_inv_4D = sympy.zeros(4, 4)
g_inv_4D[0,0] = 1/gtt_4D
g_inv_4D[1,1] = 1/grr_4D
g_inv_4D[2,2] = 1/gtheta_theta_4D
g_inv_4D[3,3] = 1/gphi_phi_4D

# Constants
R_y = sympy.symbols('R_y') # Radius of extra dimension
C = 2 * sympy.pi * R_y


# Lagrangian density (only terms relevant to dilaton and graviphoton)
L = sympy.sqrt(-gtt_4D*grr_4D*gtheta_theta_4D) * (
    -(1/2) * sympy.exp(-phi_dilaton/4) * (
        g_inv_4D[0,0] * (sympy.Derivative(phi_dilaton,t))**2 +
        g_inv_4D[1,1] * (sympy.Derivative(phi_dilaton,r))**2
        )
    - (1/4) * sympy.exp(3*phi_dilaton/4) * (
        2 * (sympy.Derivative(A_t,r))**2 * g_inv_4D[0,0] * g_inv_4D[1,1]
    )
)
L = L.subs({theta: sympy.pi/2})  # Equatorial plane

# --- Dilaton Field Equation ---
dL_dphi = L.diff(phi_dilaton)
dL_ddphi_dr = L.diff(sympy.Derivative(phi_dilaton, r))
ddr_dL_ddphi_dr = dL_ddphi_dr.diff(r)
dilaton_eq = sympy.Eq(ddr_dL_ddphi_dr - dL_dphi, 0)
dilaton_eq_simplified = sympy.simplify(dilaton_eq)
print("Dilaton Field Equation:")
print(dilaton_eq_simplified)

# Solve for the second derivative of the dilaton
ddphi_dr2_expr = sympy.solve(dilaton_eq_simplified, sympy.Derivative(phi_dilaton, (r, 2)))[0]
print("\nExpression for ddphi_dr2 (Dilaton):")
print(ddphi_dr2_expr)
ddphi_dr2_func = sympy.lambdify((r, phi_dilaton, sympy.Derivative(phi_dilaton,r), A_t, sympy.Derivative(A_t,r)), ddphi_dr2_expr)


# --- Graviphoton Field Equation ---
dL_dA_t = L.diff(A_t) # Should be zero in our simplified case
dL_ddA_t_dr = L.diff(sympy.Derivative(A_t, r))
ddr_dL_ddA_t_dr = dL_ddA_t_dr.diff(r)

graviphoton_eq = sympy.Eq(ddr_dL_ddA_t_dr - dL_dA_t, 0) # a_t equation
graviphoton_eq_simplified = sympy.simplify(graviphoton_eq)
print("\nGraviphoton Field Equation (A_t):")
print(graviphoton_eq_simplified)

# Solve for second derivative
ddA_t_dr2_expr = sympy.solve(graviphoton_eq_simplified, sympy.Derivative(A_t, (r,2)))[0]
print("\nExpression for ddA_t_dr2:")
print(ddA_t_dr2_expr)

ddA_t_dr2_func = sympy.lambdify((r, phi_dilaton, sympy.Derivative(phi_dilaton,r), A_t), ddA_t_dr2_expr)
