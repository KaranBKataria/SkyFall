"""
This script defines the ODEs for the dynamics of the satellite (EoMs) and computed the analytical
Jacobian using SymPy.

Note: SymPy must be install - this is very important since it is NOT a common Python package (c.f. NumPy, SciPy)

Module: ES98B
Group: ISEE-3
"""

import sympy as sp

# Define state variable symbols
# x, y, vx, vy = sp.symbols('x y vx vy')
r, theta, r_dot, th_dot = sp.symbols('r theta r_dot th_dot')

# Define other symbols (e.g. constants)
G, M_e, Cd, A, m, rho_b, R_star, g0, T_b, h_b, M_molar = sp.symbols('G, M_e, Cd, A, m, rho_b, R_star, g0, T_b, h_b, M_molar')
omega_E = sp.symbols('omega_E')

# Define state
# state = sp.Matrix([x, y, vx, vy])
state = sp.Matrix([r, theta, r_dot, th_dot])

# Define intermediary symbols
# r = sp.sqrt(x**2 + y**2)
# v = sp.sqrt(vx**2 + vy**2)
v_rel = sp.sqrt(r_dot**2 + (r*(th_dot - omega_E))**2)
y = r*sp.sin(theta)
rho = rho_b * sp.exp(-(g0 * M_molar * (y - h_b)) / (R_star * T_b))

# Drag coefficient
D = (1/2) * Cd * A * rho / m

# Write down acceleration due to gravity
ar_g = - G * M_e / r**2
# atheta_g = - G * M_e * y / r**2

# Write down acceleration due to drag
ar_d = -D * v_rel * r_dot + (r * th_dot**2)
atheta_d = -D * v_rel * (th_dot - omega_E) - ((2 * r_dot * th_dot)/(r))

r_dotdot = ar_g + ar_d
th_dotdot = atheta_d

# Full dynamics vector (process model)
f = sp.Matrix([
    r_dot,
    th_dot,
    r_dotdot,
    th_dotdot
])

# Compute the Jacobian (F) of the process model, f
F = f.jacobian(state)
# latex_str = sp.latex(F)
# print(latex_str)

# Enable it to be a NumPy function which can be evaluated - this will be passed into the main Predictor class in
# predictor.py
F_func = sp.lambdify((r, theta, r_dot, th_dot, G, M_e, Cd, A, m, rho_b, R_star, g0, T_b, h_b, M_molar, omega_E), F, modules='numpy')