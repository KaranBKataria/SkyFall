"""
This script implements the Simulator as a class for the 2D problem;
the code works in the Earth-Centered Inertial (ECI) frame. The
measurements provided are the radial distance (range) between the
satellite and the radar station which provided the measurement, as
well as the radial velocity.

Module: ES98B
Group: ISEE-3
"""

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

from ..utils.global_variables import *
from ..utils.predictor_utilities import *

class Simulator:

    def __init__(self, initial_state: np.array, measurement_covariance: np.array, timestep: float, t0: float = 0, t_end: float = 4e9):
        
        assert isinstance(initial_state, (list, np.ndarray)), "Initial state must be a list or a Numpy array"

        assert np.isfinite(initial_state).all() == True, "Initial state must contain finite or non-NaN values"
        assert np.isfinite(measurement_covariance).all() == True, "Measurement covariance matrix must contain finite or non-NaN values"

        assert timestep > 0.0, "Time step must be a positive float"
        assert t0 >= 0.0, "Initial time must be a non-negative float"
        assert t_end > 0.0 and t_end > t0, "Final time must be a positive float and greater than the initial time, t0"

        # Define assertion statements to ensure covariance matrices are of the correct shape
        assert measurement_covariance.shape == (2, 2), "Shape of measurement covariance matrix, R, must be (2, 2)"

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

    def get_measurements(self, verbose: bool =True) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array]:

        """
        This function produces real (noiseless) and noisy measurements (perturbed with
        additive Gaussian noise) to simulate real radar station data. This is done by
        numerically solving the EoMs using RK45, obtaining the times and states. The function
        also provides a history of which radar station's were active (i.e. provided the measurement)
        and their longitude per time step.

        Input:
                self
                verbose: A boolean expression to determine whether or not to display print statements
        
        Outputs:
                times: The times for which the ODE was numerically stepped through
                measurement_noiseless: The 'real' radar station measurements (i.e. no noise)
                measurement_noise: The radar station measurements perturbed with additive Gaussian noise
                active_radar_stations_per_time: A history of active radar stations per time step by index
                active_radar_station_longitudes_per_time: A history of active radar stations' longitudes per time
        """

        if verbose is True:
            print("Initialising simulator.")

        t0 = self.t0
        t_end = self.t_end
        t_eval = self.t_eval
        timestep = self.timestep
        state = np.asarray(self.initial_state)
        measurement_covariance_matrix = np.asarray(self.measurement_covariance)

        half_beam = np.deg2rad(60/2)
        beam_width = 2 * half_beam

        # Define the number of radar stations based on beam width
        N_stn = int(np.ceil(2*np.pi/beam_width))

        # Define initial latitudes of each of the N_stn radar stations 
        theta_R0 = np.linspace(0, 2*np.pi, N_stn, endpoint=False)

        if verbose is True:
            print("Generating radar measurements...")

        # Use RK45 to numerically solve the ODEs (EoMs)
        sol = solve_ivp(
            equations_of_motion,
            (t0, t_end),
            state,
            method='RK45',
            t_eval=t_eval,
            events=hit_ground,
            rtol=1e-8,
            atol=1e-8
            #max_step=timestep
        )

        # Extract the solutions
        times, state_noiseless, crash_site, crash_time = sol.t, sol.y, sol.y_events[0].flatten(), sol.t_events[0].flatten()

        # Define lists to collect clean and noisy measurements
        measurement_noiseless = []
        measurement_noise = []

        # Define lists to collect active radar station indicies and longitudes
        active_radar_stations_per_time = []
        active_radar_station_longitudes_per_time = []
        
        # For each state and time, obtain the noisy and noiseless measurements
        for index, (time, state) in tqdm(enumerate(zip(times, state_noiseless.T)), desc="Finalising radar measurements..."):

            # Extract states
            r, theta, r_dot, th_dot = state

            # Compute radar station longitudes in ECI at time t ([-pi, pi])
            theta_R = (theta_R0 + omega_E *  time - np.pi) % (2*np.pi) - np.pi

            # [-pi, pi]
            dth = ((theta - theta_R + np.pi) % (2*np.pi)) - np.pi

            # Obtain index of the radar station with the closest beam/measurement to satellite
            index_active_radar = np.argmin(np.abs(dth)) 
            active_radar_stations_per_time.append(index_active_radar)

            # Convert the ODE solutions to active radar station measurement

            # [0, 2pi] 
            active_radar_longitude = theta_R[index_active_radar]
            radar_measurements_noiseless = measurement_model_h(state=state, radar_longitude=active_radar_longitude)
            
            active_radar_station_longitudes_per_time.append(active_radar_longitude)

            # Perturb the 'real' radar measurements with Gaussian noise 
            gaussian_noise = np.random.multivariate_normal(
                mean=np.zeros(radar_measurements_noiseless.size),
                cov=measurement_covariance_matrix,
                size=1).T

            radar_measurements_noise = radar_measurements_noiseless + gaussian_noise.flatten()

            measurement_noiseless.append(radar_measurements_noiseless)
            measurement_noise.append(radar_measurements_noise)
        
        measurement_noiseless = np.asarray(measurement_noiseless)
        measurement_noise = np.asarray(measurement_noise)
        active_radar_stations_per_time = np.asarray(active_radar_stations_per_time)
        active_radar_station_longitudes_per_time = np.asarray(active_radar_station_longitudes_per_time)

        if verbose is True:
            print("Terminating simulator; outputting radar measurements and additional data.")

        return times, measurement_noiseless, measurement_noise, active_radar_stations_per_time, active_radar_station_longitudes_per_time, crash_site, crash_time[0], state_noiseless.T
