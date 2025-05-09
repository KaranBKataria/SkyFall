"""
This script implements the Simulator for the 1D problem where
there is a flat Earth with acceleration due to gravity only in the y-component
going straight down.

Module: ES98B
Group: ISEE-3
"""

import numpy as np
from scipy.integrate import solve_ivp

from ..utils.global_variables import *
from ..utils.predictor_utilities import *

class Simulator:

    def __init__(self, initial_state: np.array, measurement_covariance: np.array, timestep: float, t0: float, t_end: float = 20000.0):
        
        # Define an attribute for the measurement covariance matrix R, which will be used to sample a multivariate
        # Gaussian distribution to perturb the radar measurements with noise

        # NB: The SAME R is used in the Simulator and the Predictor
        self.measurement_covariance = measurement_covariance

        # Define attributes for times, timesteps and time ranges when solving the ODE numerically using RK45
        self.timestep = timestep
        self.t0 = t0
        self.t_end = t_end
        self.t_eval = np.arange(t0, t_end+timestep, timestep)

        # Define an attribute for the initial state
        self.initial_state = initial_state

    def get_measurements(self):

        """
        This function produces real (noiseless) and noisy measurements (perturbed with
        additive Gaussian noise) to simulate real radar station data. This is done by
        numerically solving the EoMs using RK45, obtaining the times and states.

        Input:
                self
        
        Outputs:
                times: The times for which the ODE was numerically stepped through
                state_noiseless: The 'real' radar station measurements (i.e. no noise)
                state_noise: The radar station measurements perturbed with additive Gaussian noise
        """

        t0 = self.t0
        t_end = self.t_end
        t_eval = self.t_eval
        state = np.asarray(self.initial_state)
        measurement_covariance_matrix = np.asarray(self.measurement_covariance)

        # Use RK45 to numerically solve the ODEs (EoMs)
        sol = solve_ivp(
            equations_of_motion,
            (t0, t_end),
            #[x0, y0, vx0, vy0],
            state,
            method='RK45',
            t_eval=t_eval,
            events=hit_ground,
            rtol=1e-8,
            atol=1e-8
        )

        # Extract the solutions
        times, state_noiseless = sol.t, sol.y

        # Obtain the dims and sample from the zero-mean Gaussian distribution using measurement covariance matrix R
        dim = times.size

        gaussian_noise = np.random.multivariate_normal(mean=np.zeros(state.size), cov=measurement_covariance_matrix, size=dim).T

        # Perturb the 'real' radar measurements with Gaussian noise 
        state_noise = state_noiseless + gaussian_noise

        return times, state_noiseless.T, state_noise.T


# # # Measurement Noise: depends on satelillte position from radar station
# sigma_y0 = 10

# # Time grid
# t0 = 0
# t_end = 20000.0 # any large number doesn't really matter 
# dt = 1.0 # change the timestep depending on typical times 
# t_eval = np.arange(t0,t_end+dt, dt)

# # Initial State
# x0  = 0.0 # starting directly above radar station
# y0 = 600e3 # distance from earth's surface
# vx0 = 100e3 # horizontal speed
# vy0 = 0 # vertical speed

# # radar coverage
# half_beam = np.deg2rad(60.0/2.0)

# # Length of flat earth is the circumference of Earth as a flat line
# L = 2 * np.pi * R_e

# # Width of coverage at initial position
# initial_width = 2.0 * y0 *np.tan(half_beam)

# N_stations = int(np.ceil(L/initial_width))
# spacing = L/N_stations
# stations = np.arange(N_stations)*spacing


# # wrapping x to [0,L]
# # did this to make sure that horizontal coordinate is on the earth's surface
# # falls within one orbit
# x_wrapped = xs % L

# # measurement with switching radar
# y_meas = np.empty_like(ys)
# # nearest radar station in order to track satellite
# # then adding fixed noise (10m) to measurement 
# for i, y in enumerate(ys):
#     idx = int(x_wrapped[i] // spacing)
#     y_meas[i] = y + np.random.normal(0, sigma_y0)

# if __name__ == "__main__":
#     for t, x, y, vx, vy, ym in zip(times, xs, ys, vxs, vys, y_meas):
#         print(f"t={t:7.1f} s | "
#             f"x={x:9.1f} m | y={y:8.1f} m | "
#             f"vx={vx:7.1f} m/s | vy={vy:7.1f} m/s | "
#             f"y_meas={ym:8.1f} m")

#     if sol.t_events[0].size > 0:
#         print(f"\nImpact at t = {sol.t_events[0][0]:.1f} s → y=0 reached.")

#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(10,6))

#     cmap = plt.get_cmap("tab20")
#     N = len(stations)
#     # Radar beam coverage
#     y_line = np.linspace(0, ys.max(), 100)
#     for i, x_stat in enumerate(stations):
#         c = cmap(i / N)
#         # left edge
#         plt.plot(x_stat - y_line/np.tan(half_beam),
#              y_line,
#              linestyle="--",
#              color=c,
#              linewidth=0.8)
#         # right edge
#         plt.plot(x_stat + y_line/np.tan(half_beam),
#              y_line,
#              linestyle="--",
#              color=c,
#              linewidth=0.8)

#     plt.plot(x_wrapped, ys, color='k', lw=2, label='Satellite trajectory')
#     plt.xlim(0, L)
#     plt.ylim(0, ys.max()*1.05)
#     plt.xlabel('x mod L (m)')
#     plt.ylabel('y (m)')
#     plt.title('Satellite trajectory and radar station beam coverage')
#     plt.legend(loc='upper right')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

    # x0 =[0.0, 600e3, 100e3, 0]
    # R = covariance_matrix_initialiser(variances=[10, 10, 10, 10])
    # del_t = 1.0
    # t0 = 0

    # sim = Simulator(initial_state=x0, measurement_covariance=R, timestep=del_t, t0=t0)


