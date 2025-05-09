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


# Atmosphere constants (https://en.wikipedia.org/wiki/Barometric_formula)

# Molar mass of Earth's air in kilograms per mole 
M_molar = 0.0289644 

# Mass density (kg/m^3) 
rho0 = 1.225  

# Scale height
H = 8400.0

# Universal gas constant
R_star = 8.3144598

# Atmosphere layers; the values used by the Barometric formula will change with altitude y (m)
layers = [
    {"h":     0.0, "rho":1.22500,   "T":288.15},
    {"h": 11000.0, "rho":0.36391,   "T":216.65},
    {"h": 20000.0, "rho":0.08803,   "T":216.65},
    {"h": 32000.0, "rho":0.01322,   "T":228.65},
    {"h": 47000.0, "rho":0.00143,   "T":270.65},
    {"h": 51000.0, "rho":0.00086,   "T":270.65},
    {"h": 71000.0, "rho":0.000064,  "T":214.65},
]


# ISEE-3 satellite constants (https://en.wikipedia.org/wiki/International_Cometary_Explorer)

# Mass of ISEE-3 satellite in kilograms 
m_s = 479  

# Drag coefficient
C_d = 2.2

# Dimensions
r = 1.77/2

# Compute area of the ISEE-3 satellite
# A = 2*np.pi*(r)*1.58
A = np.pi * r**2 # front area of the satellite, check with peter to see if this is correct
