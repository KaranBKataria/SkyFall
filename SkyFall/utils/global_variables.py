"""
This file contains all global (shared) constants that will be
used across multiple files in the project (e.g. simulator,
predictor etc).

Module: ES98B
Group: ISEE-3
"""

import numpy as np

# Physical constants 

# Gravitational constant
G = 6.67430e-11

# Mass of the Earth in kilograms
M_e = 5.972e24 

# Radius of the Earth in metres
R_e = 6.371e6 

# Acceleration due to Earth's gravity in metres per second
g0 = 9.80665 

# Angular frequency of the Earth's rotation
omega_E = 7.2921150e-5

# Atmosphere constants (https://en.wikipedia.org/wiki/Barometric_formula)

# Molar mass of Earth's air in kilograms per mole 
M_molar = 0.0289644 

# Mass density (kg/m^3) 
rho0 = 1.225  

# Scale height
H = 8400.0

# Universal gas constant
R_star = 8.3144598

# Specific Gas Constant
R_air = 287.05287

# US Standard atmospheric model layers
layers = [ # H_b, T_b, L_b           
    (0.0,       288.15, -6.5e-3),
    (11e3,      216.65,  0.0   ),
    (20e3,      216.65,  1.0e-3),
    (32e3,      228.65,  2.8e-3),
    (47e3,      270.65,  0.0   ),
    (51e3,      270.65, -2.8e-3),
    (71e3,      214.65, -2.0e-3),
    (84.852e3,  186.867, 0.0   ),
]

# kg/m^3 at sea level
rho0 = 1.225        

# Compute a list of base densities at each layer             
base_rho = [rho0]
for k in range(1, len(layers)):
    H_b, T_b, L_b = layers[k-1]
    H_t           = layers[k][0]
    dh            = H_t - H_b
    if L_b != 0.0:
        fac = (T_b / (T_b + L_b*dh))**(1 + g0/(R_air*L_b))
    else:
        fac = np.exp(-g0*dh/(R_air*T_b))
    base_rho.append(base_rho[-1] * fac)

# ISEE-3 satellite constants (https://en.wikipedia.org/wiki/International_Cometary_Explorer)

# Mass of ISEE-3 satellite in kilograms 
m_s = 479

# Drag coefficient
C_d = 2.2

# Radius of cross-sectional area (satellite assumed to be cylindrical)
r = 1.77/2

# Compute cross-sectional area of the ISEE-3 satellite
# A = 2*np.pi*(r)*1.58
A = np.pi * r**2 # front area of the satellite, check with peter to see if this is correct
