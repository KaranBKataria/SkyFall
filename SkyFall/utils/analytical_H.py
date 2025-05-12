"""
This script defines the measurement model for the EKF and computes the analytical
Jacobian using SymPy. This model is framed in the ECI coordinate system.

Note: SymPy must be installed - this is very important since it is NOT a common Python package (c.f. NumPy, SciPy)

Module: ES98B
Group: ISEE-3
"""

import sympy as sp

# Define state variable symbols
r, theta, r_dot, th_dot = sp.symbols('r theta r_dot th_dot')

# Define other symbols (e.g. constants)
R_e, omega_E, theta_R = sp.symbols('R_e omega_E theta_R')

# Define state
state = sp.Matrix([r, theta, r_dot, th_dot])

# Define intermediary symbols

# Velocity of satellite
v_s = sp.Matrix([
    r_dot * sp.cos(theta) - r * th_dot * sp.sin(theta),
    r_dot * sp.sin(theta) + r * th_dot * sp.cos(theta)
])

# Velocity of the radar station
v_r = sp.Matrix([
    -R_e * omega_E * sp.sin(theta_R),
    R_e * omega_E * sp.cos(theta_R)
])

# Cartesian position of satellite
r_s = sp.Matrix([
    r * sp.cos(theta),
    r * sp.sin(theta)
])

# Cartesian position of the radar station 
r_r = sp.Matrix([
    R_e * sp.cos(theta_R),
    R_e * sp.sin(theta_R)
])

# Position vector between radar station and satellite
rad_dist_vector = r_s - r_r

# Velocity vector between radar station and satellite
rad_vel_vector = v_s - v_r

# Radial distance between radar station and satellite 
rad_dist = sp.sqrt((r * sp.cos(theta) - R_e * sp.cos(theta_R))**2 + (r * sp.sin(theta) - R_e * sp.sin(theta_R))**2)

# Radial velcoity between radar station and satellite
rad_vel = rad_vel_vector.T * ((rad_dist_vector)/(rad_dist))

# Full measurement vector (measurement model)
h = sp.Matrix([
    rad_dist,
    rad_vel
])

# Compute the Jacobian (F) of the process model, f
H = h.jacobian(state)
# latex_str = sp.latex(H)
# print(latex_str)

# Enable it to be a NumPy function which can be evaluated - this will be passed into the main Predictor class in
# predictor.py
H_func = sp.lambdify((r, theta, r_dot, th_dot, R_e, omega_E, theta_R), H, modules='numpy')