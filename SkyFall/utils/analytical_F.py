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
G, M_e, Cd, A, m, rho_b, R_air, g0, T_b, h_b, h_s, L_b, R_e, omega_E = sp.symbols('G M_e Cd A m rho_b R_air g0 T_b h_b h_s L_b R_e omega_E')

# Define state
# state = sp.Matrix([x, y, vx, vy])
state = sp.Matrix([r, theta, r_dot, th_dot])

# Define intermediary symbols
# r = sp.sqrt(x**2 + y**2)
# v = sp.sqrt(vx**2 + vy**2)
v_rel = sp.sqrt(r_dot**2 + (r*(th_dot - omega_E))**2)
h = r - R_e

# Drag coefficient and density
rho1 = rho_b * (T_b / (T_b + (L_b*(h - h_b)))) ** (1 + ((g0)/(R_air * L_b)))
rho2 = rho_b * sp.exp(-((g0 * (h - h_b)) / (R_air * T_b)))
rho3 = rho_b * sp.exp(-((h - h_b) / (h_s)))

D1 = ((1/2) * Cd * A * rho1) / (m)
D2 = ((1/2) * Cd * A * rho2) / (m)
D3 = ((1/2) * Cd * A * rho3) / (m)

# Write down acceleration due to gravity
ar_g = (- G * M_e) / (r**2)
# atheta_g = - G * M_e * y / r**2

# Write down acceleration due to drag
ar_d1 = -D1 * v_rel * r_dot + (r * th_dot**2)
atheta_d1 = (-D1 * v_rel * r * (th_dot - omega_E) - (2 * r_dot * th_dot))/(r)

ar_d2 = -D2 * v_rel * r_dot + (r * th_dot**2)
atheta_d2 = (-D2 * v_rel * r * (th_dot - omega_E) - (2 * r_dot * th_dot))/(r)

ar_d3 = -D3 * v_rel * r_dot + (r * th_dot**2)
atheta_d3 = (-D3 * v_rel * r * (th_dot - omega_E) - (2 * r_dot * th_dot))/(r)

r_dotdot1 = ar_g + ar_d1
th_dotdot1 = atheta_d1

r_dotdot2 = ar_g + ar_d2
th_dotdot2 = atheta_d2

r_dotdot3 = ar_g + ar_d3
th_dotdot3 = atheta_d3

# Full dynamics vector (process model)
f1 = sp.Matrix([
    r_dot,
    th_dot,
    r_dotdot1,
    th_dotdot1
])

f2 = sp.Matrix([
    r_dot,
    th_dot,
    r_dotdot2,
    th_dotdot2
])

f3 = sp.Matrix([
    r_dot,
    th_dot,
    r_dotdot3,
    th_dotdot3
])

# Compute the Jacobian (F) of the process model, f
F1 = f1.jacobian(state)
F2 = f2.jacobian(state)
F3 = f3.jacobian(state)

# latex_str = sp.latex(F)
# print(latex_str)

# Enable it to be a NumPy function which can be evaluated - this will be passed into the main Predictor class in
# predictor.py
F_func1 = sp.lambdify((r, theta, r_dot, th_dot, G, M_e, Cd, A, m, rho_b, R_air, g0, T_b, h_b, L_b, R_e, omega_E), F1, modules='numpy')
F_func2 = sp.lambdify((r, theta, r_dot, th_dot, G, M_e, Cd, A, m, rho_b, R_air, g0, T_b, h_b, R_e, omega_E), F2, modules='numpy')
F_func3 = sp.lambdify((r, theta, r_dot, th_dot, G, M_e, Cd, A, m, rho_b, R_air, g0, h_b, h_s, R_e, omega_E), F3, modules='numpy')