"""
This script defines the main predictor class, containing the key functionality
the user will interact with via methods. See a toy example at the bottom making use
of this class.

Module: ES98B
Group: ISEE-3
"""

# Import in the key functionalities; NB: addititional modules will also be loaded that are imported into the predictor_utilities
# file. Ask Karan for more info. if needed
from SkyFall.utils import *

class Predictor:

    """
    This class defines the key functionalities of the predictor, including the EKF algorithm
    and the crash site/time forecasting functionality.
    """

    def __init__(
            self, process_covariance, measurement_covariance, state_covariance, initial_state, timestep, t0):
        
        assert isinstance(initial_state, (list, np.ndarray)), "Initial state must be a list or a Numpy array"
        initial_state = np.asarray(initial_state)

        assert np.isfinite(initial_state).all() == True, "Initial state must contain finite or non-NaN values"
        assert np.isfinite(process_covariance).all() == True, "Process covariance matrix must contain finite or non-NaN values"
        assert np.isfinite(measurement_covariance).all() == True, "Measurement covariance matrix must contain finite or non-NaN values"
        assert np.isfinite(state_covariance).all() == True, "State covariance matrix must contain finite or non-NaN values"

        assert timestep > 0, "Time step must be a positive float"
        assert t0 >= 0, "Initial time must be a non-negative float"

        # Define assertion statements to ensure covariance matrices are of the correct shape
        assert process_covariance.shape == (4, 4), "Shape of process covariance matrix, Q, must be (4, 4)"
        assert measurement_covariance.shape == (2, 2), "Shape of measurement covariance matrix, R, must be (2, 2)"
        assert state_covariance.shape == (4, 4), "Shape of state covariance matrix, P, must be (4, 4)"

        # Define assert statement to ensure state vector is of the correct shape
        assert initial_state.shape == (4,), "Shape of the initial state must be (4,)"

        # Define the covariance matrices. NB: There will be 3 seperate states the state vector and state covariance matrix
        # can take.
        self.process_covariance = process_covariance
        self.measurement_covariance = measurement_covariance
        
        self.initial_state_covariance = state_covariance
        self.prior_state_covariance = state_covariance
        self.posterior_state_covariance = state_covariance

        self.initial_state = initial_state
        self.prior_state = initial_state
        self.posterior_state = initial_state

        # Track the posterior 
        self.prior_traj = [initial_state]
        self.posterior_traj = [initial_state]
        self.posterior_traj_states_LLA = [ECI_to_ECEF(time=t0, state=initial_state)[-1]]
        self.posterior_traj_states_cartesian = [ECI_to_ECEF(time=t0, state=initial_state)[-3]]
        self.posterior_traj_times = [t0]

        # Define the timesteps, initial time and a time variable t to track the evolution of the predictor
        self.timestep = timestep
        self.t0 = t0
        self.t = t0

        # Initially instantiated to None; this will be replaced in every iteration
        self.kalman_gain_matrix = None
        self.res = None
        self.JacobianH = None
        self.JacobianF = None
        
        # Define attributes to collect the forecasted times and states at each iteration, as well as their statistics
        self.forecasted_states = []
        self.forecasted_states_mean = []
        self.forecasted_states_std = []

        self.forecasted_states_LLA_deg = []
        self.forecasted_states_LLA_deg_mean = []
        self.forecasted_states_LLA_deg_std = []

        self.forecasted_times = []
        self.forecasted_times_mean = []
        self.forecasted_times_std = []

    def process_model(self, include_noise=True, verbose: bool = True):
        """
        This function outputs a prior state estimate (by default with no additive Gaussian noise) under the
        non-linear process model given by the equations of motion at the next time step.

            Inputs:
                    self.posterior_state: the previous, data assimilated state
                    self.process_covariance: the process covariance matrix
                    self.timestep: the timestep for the RK23 ODE solver
                    self.t: previous time of the system
                    include_noise: a boolean arguement to determine whether or not to perturb state with Gaussian noise
                    verbose: boolean arguement to output printed statements or not

            Outputs:
                    predicted_state_with_noise: the prior state estimate before data assimilation, perturbed with noise
                    predicted_state_without_noise: the prior state estimate before data assimilation, with no noise
        """

        state = np.asarray(self.posterior_state)

        # Determine the cardinality of the state vector and ensure process covariance matrix is an array
        xdim = state.size

        Q = np.asarray(self.process_covariance)

        timestep = self.timestep
        t = self.t

        # Ensure we are updating the global time variable
        t_eval = np.arange(t, t + 2*timestep, timestep)

        # Numerically solve the ODE up to the next time step
        predicted_state = solve_ivp(
            fun=equations_of_motion,
            t_span=(t, t + timestep),
            y0=state,
            method='RK45',
            t_eval=t_eval,
            events=hit_ground,
            rtol=1e-8,
            atol=1e-8
        )

        # Update global time variable
        self.t += timestep

        # If noise is to be included, perturb the state with additive Gaussian noise - else return just the predicted state
        if include_noise is True:

            # Sample from a multivariate, zero-mean Gaussian distribution with the process covariance matrix 
            additive_gaussian_noise = np.random.multivariate_normal(mean=np.zeros(xdim), cov=Q, size=1).flatten()
            predicted_state_with_noise = np.asarray(predicted_state.y)[:,-1] + additive_gaussian_noise

            self.prior_state = predicted_state_with_noise
            self.prior_traj.append(predicted_state_with_noise)
        
        else:
            predicted_state_without_noise = np.asarray(predicted_state.y)[:,-1]
            
            self.prior_state = predicted_state_without_noise
            self.prior_traj.append(predicted_state_without_noise)

        if verbose is True:
            print(f'Current prior state:\n {self.prior_state}\n')
    
    def measurement_model(self, theta_R: float) -> np.array:
        """
        This function returns the evaluate of the measurement model h
        at the prior state estimate as part of the EKF algorithm.

            Input:
                    self
                    theta_R: the longitude of the 'active' radar at time self.t
            
            Output:
                    h: the measurement model output
        """

        prior_state = np.asarray(self.prior_state)

        h = measurement_model_h(prior_state, radar_longitude=theta_R)

        return h

    def eval_JacobianF(self, G=G, M_e=M_e, Cd=C_d, A=A, m=m_s, R_air=R_air, g0=g0, omega_E=omega_E, R_e=R_e, h_s=h_s, verbose=False) -> np.array:

        """
        This function evaluates the analytical Jacobian of process model using SymPy. This function
        (F_func) is imported from the analytical_F.py file and is evaluated on the posterior state
        from the previous time step.

            Inputs:
                    G: Graviational constant
                    M_e: Mass of the Earth (kg)
                    Cd: Drag coefficient of the satellite
                    A: Cross-sectional area of the satellite (m^2)
                    m: Mass of the satellite (kg)
                    R_star: Universal gas constant
                    g0: Acceleration due to Earth's gravity (m/s^2)
                    M_molar: Molar mass of Earth's air in kilograms per mole
                    omega_E: Angular velocity of the Earth

            Outputs:
                    self.JacobianF: The Jacobian evaluated at the posterior state of the
                                    previous time step
        """
    
        # Ensure posterior state of previous time step is an array and extract it's components
        state = np.asarray(self.posterior_state)
        r, theta, r_dot, th_dot = state

        # Select correct parameters of the Barometric formula based on altitude (r)
        altitude = max(r - R_e, 0.0)

        if altitude > 86e3:

            ds = ussa1976.compute( np.linspace( (altitude/1000)-1, (altitude/1000)+1, num=3) )
            rhos=ds["rho"].values
            rho3=rhos[1]

            F = F_func3(r=r, theta=theta, r_dot=r_dot, th_dot=th_dot, 
                        G=G, M_e=M_e, Cd=Cd, A=A, m=m, omega_E=omega_E, rho3=rho3)
                        
            self.JacobianF = F
        
        # If altitude is less than 86km
        elif altitude < 86e3:

            # Obtain layer constants depending on altitude
            for index, (h_b, _, _) in enumerate(layers):
                if altitude >= h_b:
                    layer = index
            
            h_b, T_b, L_b = layers[layer]
            rho_b = base_rho[layer]

            # If the Lapse rate is non-zero, obtain Jacobian of f via F_func1
            if L_b != 0.0:
                
                F = F_func1(
                    r=r, theta=theta, r_dot=r_dot, th_dot=th_dot,
                    G=G, M_e=M_e, Cd=Cd, A=A, m=m, rho_b=rho_b, R_air=R_air,
                    g0=g0, T_b=T_b, h_b=h_b, L_b=L_b, R_e=R_e, omega_E=omega_E)
                
                self.JacobianF = F

            # Else obtain it via F_func2
            else:
                
                F = F_func2(
                    r=r, theta=theta, r_dot=r_dot, th_dot=th_dot,
                    G=G, M_e=M_e, Cd=Cd, A=A, m=m, rho_b=rho_b, R_air=R_air,
                    g0=g0, T_b=T_b, h_b=h_b, R_e=R_e, omega_E=omega_E)

                self.JacobianF = F
            
        if verbose is True:
            print(f'The Jacobian F: {self.JacobianF}')

    def eval_JacobianH(self, theta_R: float, R_e=R_e, omega_E=omega_E) -> np.array:
        """
        This function evaluates the Jacobian of the measurement process at the current
        prior state.

            Inputs:
                    theta_R: the longitude of the 'active' satellite at time self.t
                    R_e: the radius of the Earth
                    omega_E: the angular velocity of the Earth

            Outputs:
                    self.JacobianH: the evaluated Jacobian of the measurement model
        """

        # Ensure posterior state of previous time step is an array and extract it's components
        state = np.asarray(self.posterior_state)
        r, theta, r_dot, th_dot = state
        
        # Evaluate you the Jacobian of the measurement model
        H = H_func(r=r, theta=theta, r_dot=r_dot, th_dot=th_dot, R_e=R_e, omega_E=omega_E, theta_R=theta_R)

        # n_dims = np.asarray(self.prior_state).size
        self.JacobianH = H

    def update_prior_belief(
        self, JacobianV=None, control_covariance=None, verbose: bool = True):
        """
        This function updates the state covariance matrix P for the next iteration in the
        EKF. Note, this is the state covariance matrix BEFORE data assimilation - i.e. it is
        P bar according to the notation. See function "assimilated_posterior_prediction" for
        the posterior state covariance matrix after taking into account the measurement data.

            Inputs:
                    self.JacobianF: the jacobian of the process model evaluated at the previous state
                    self.process_covariance: the process covariance matrix
                    self.posterior_state_covariance: the now prior state covariance matrix of the previous state 
                    JacobianV: the jacobian of the control model evaluated at the previous control state
                    control_covariance: the control covariance matrix
                    verbose: boolean arguement to output printed statements or not
            
            Outputs:
                    P_bar: the updated process covariance matrix before data assimilation (P bar
                                according to the mathematical notation)

            Notes:
                    Control process is optional; if not provided the update proceeds as though there is no
                    control model.
        """
        # Create local variables to have consistent notation as the mathematical notation
        Q = np.asarray(self.process_covariance)
        F = np.asarray(self.JacobianF)
        P = np.asarray(self.posterior_state_covariance)

        # If control input is provided, compute posterior state covariance taking this into account
        if JacobianV is not None:
            V = np.asarray(JacobianV)
            M = np.asarray(control_covariance)
            
            P_bar = (F @ P @ F.T) + (V @ M @ V.T) + Q
            P_bar = 0.5 * (P_bar + P_bar.T)
            self.prior_state_covariance = P_bar
        
        # Else, update the state covariance without any control inputs
        else: 
            P_bar = (F @ P @ F.T) + Q
            P_bar = 0.5 * (P_bar + P_bar.T)
            self.prior_state_covariance = P_bar

        if verbose is True:
            print(f'Current prior state covariance matrix:\n {self.prior_state_covariance}\n')

    def residual(self, measurement: np.array, theta_R: float, verbose: bool = True):
        """
        This function computes the residual of the true measurement data and the predicted
        measurement data, computed from the measurement model and the prior state estimate 
        before data assimilation.

            Inputs:
                    measurement: the measurement data from the simulator at the current time
                    measurement_model: the measurement model function
                    self.prior_state: the prior state estimate at the current time, before data assimilation
                    verbose: boolean arguement to output printed statements or not
            
            Output:
                    res: the residual value, y
        """
        
        residual = np.array(measurement - self.measurement_model(theta_R=theta_R))

        # If there is a single value (i.e. size-1 array), reshape it for compatibility with later linear algebra
        if residual.size == 1:
            residual = residual.reshape(1,1)

        self.res = residual

        if verbose is True:
            print(f'Residual value:\n {self.res}\n')

    def kalman_gain(self, verbose: bool = True):
        """
        This function computes the Kalman gain matrix.

            Inputs:
                    self.prior_state_covarance: the updated state covariance matrix (P bar) before data assimilation 
                    self.measurement_covariance: the measurement covariance matrix
                    self.JacobianH: the Jacobian of the measurement model evaluated at the predicted state before data assimilation
                    verbose: boolean arguement to output printed statements or not
            
            Outputs:
                    K: the Kalman gain matrix

            Notes:
                    To prevent numerical instability arising from matrix inversion, a linear solve has been performed instead
                    to compute the Kalman gain matrix. The inversion matrix (S below) may have a low condition number if
                    evaluated but nevertheless a linear solve was performed to prevent any ill-conditioning.
        """

        # Create local variables to have consistent notation as the mathematical notation
        P_bar = np.asarray(self.prior_state_covariance) 
        R = np.asarray(self.measurement_covariance)
        H = np.asarray(self.JacobianH)

        # Ensure arrays have the correct shape for the linear algebra
        if P_bar.ndim == 1:
            P_bar = P_bar.reshape(1, 1)
        elif R.ndim == 1:
            R = R.reshape(1, 1)
        elif H.ndim == 1:
            H = H.reshape(1, 1)
                
        # Create intermediate variable
        S = H @ P_bar @ H.T + R

        # Compute the Kalman Gain using linear solve (not inversion due to numerical instability)
        K = np.linalg.solve(S, (P_bar @ H.T).T).T

        self.kalman_gain_matrix = K

        if verbose is True:
            print(f'Current Kalman gain:\n {self.kalman_gain_matrix}\n')

    def assimilated_posterior_prediction(self, verbose: bool = True):
        """
        This function computes the data assimilated state prediction and updated process
        covariance matrix P.

            Inputs:
                    self.kalman_gain_matrix: the Kalman gain matrix
                    self.JacobianH: the Jacobian of the measurement model evaluated at the predicted state before data assimilation
                    self.prior_state_covarance: the updated state covariance matrix (P bar) before data assimilation 
                    self.res: the residual (a.k.a the innovation) between the actual and predicted measurement data 
                    self.prior_state: the prior predicted state before assimilating any measurement data
                    verbose: boolean arguement to output printed statements or not
            
            Outputs:
                    x_state_assimilated: the assimilated, final prediction of the state at the current time 
                    P_assimilated: the data assimilated, final process covariance matrix
        """

        # Create local variables to have consistent notation as the mathematical notation
        K = np.asarray(self.kalman_gain_matrix)
        H = np.asarray(self.JacobianH)
        P_bar = np.asarray(self.prior_state_covariance)
        y = np.asarray(self.res)
        x_bar = np.asarray(self.prior_state)

        # Ensure arrays have the correct shape for the linear algebra
        if K.ndim == 1:
            K = K.reshape(1,1)
        elif H.ndim == 1:
            H = H.reshape(1, 1)
        elif P_bar.ndim == 1:
            P_bar = P_bar.reshape(1, 1)
        elif y.size == 1:
            y = y.reshape(1, 1)

        # Compute the posterior (data assimilated) state covariance matrix
        KH = K @ H
        dims = KH.shape[0]  # Determine dimensions to ensure dimensions of identity matrix are correct 
        I = np.identity(n=dims)
        P_assimilated = (I - KH) @ P_bar
        P_assimilated = 0.5 * (P_assimilated + P_assimilated.T)

        self.posterior_state_covariance = P_assimilated

        # Compute the posterior (data assimilated) state vector
        x_state_assimilated = x_bar + (K @ y)
        self.posterior_state = x_state_assimilated

        if verbose is True:
            print(f'Current posterior state:\n {self.posterior_state}\n')
            print(f'Current posterior state covariance matrix:\n {self.posterior_state_covariance}\n')

        # Append to posterior trajectories list in Cartesian coordinates (for visualisation purposes)
        self.posterior_traj.append(x_state_assimilated)
        self.posterior_traj_states_LLA.append(ECI_to_ECEF(time=self.t, state=x_state_assimilated)[-1])
        self.posterior_traj_states_cartesian.append(ECI_to_ECEF(time=self.t, state=x_state_assimilated)[-3])
        self.posterior_traj_times.append(self.t)

    def forecast(self, n_samples: int, final_time=4e9, verbose: bool = True):
        """
        This function, at each given time step after the data has been assimilated, forecasts the crash site and timing.
        Sampling-based (a.k.a Monte Carlo) uncertainty propogation has been used to obtain a distribution of the crash
        site predictions and timings (instead of point estimates).

            Inputs:
                    n_samples: the number of Monte Carlo samples to draw from the multivariate Gaussian state distribution
                    self.posterior_state_covariance: the updated state covariance matrix after data assimilation
                    self.posterior_state: the updated state vector after data assimilation
                    final_time: the final time the RK23 solver should solve up to; select a large time value as the
                                termination of the solver will happen sooner
                    verbose: boolean arguement to output printed statements or not

            Outputs:
                    predictions: the distribution of predictions; containing n_samples state vectors
                    crash_times: the distribution of crash times; containing n_samples state vectors
        """

        # Ensure covariance matrix and state vector are numpy arrays
        state = np.asarray(self.posterior_state)
        state_covariance = np.asarray(self.posterior_state_covariance)
        state_covariance += 1e-6 * np.eye(state_covariance.shape[0])

        # Ensure posterior state covariance matrix is valid (symmetric and PSD)
        assert np.allclose(state_covariance, state_covariance.T, atol=1e-10), "Posterior covariance matrix is not symmetric"
        # assert np.min(np.linalg.eigvalsh(state_covariance)) >= 0, "Posterior covariance matrix is not PSD"

        #  Extract CURRENT time and timestep
        t = self.t
        timestep = self.timestep

        # Define lists to be populated with sample crash predictions and timings
        predictions = []
        predictions_LLA_deg = []
        crash_times = []

        # if samples.size == 0:
        #     raise AssertionError("No samples drawn from Gaussian distribution; invalid covariance matrix")

        t_eval = np.arange(t, final_time+timestep, timestep)

        # For each sample, solve the ODE system to obtain a distribution - this is sampling-based uncertainty propogation
        for _ in range(n_samples):
            max_tries = 100
            
            for iter in range(max_tries):
                sample = np.random.multivariate_normal(mean=state, cov=state_covariance)

                forecasted_state = solve_ivp(
                    fun=equations_of_motion,
                    t_span=(t, final_time),
                    y0=sample,
                    method='RK45',
                    t_eval=t_eval,
                    events=hit_ground,
                    rtol=1e-8,
                    atol=1e-8
                )

                if forecasted_state.y_events[0].size > 0:
                    crash_state = forecasted_state.y_events[0].flatten()
                    predictions.append(crash_state)
                    predictions_LLA_deg.append(ECI_to_ECEF(time=self.t, state=crash_state)[-1])
                    crash_times.append(forecasted_state.t_events[0] + self.t)
                    break

        # Return the distribution of predictions and timings, and their statistics, reshaped into an appropriate format
        predictions = np.array(predictions).reshape(n_samples, state.shape[0])
        predictions_LLA_deg = np.array(predictions_LLA_deg).reshape(n_samples, state.shape[0]-1)
        crash_times = np.array(crash_times).reshape(n_samples, 1)

        self.forecasted_states.append(predictions)
        self.forecasted_states_mean.append(predictions.mean(axis=0))
        self.forecasted_states_std.append(predictions.std(axis=0))

        self.forecasted_states_LLA_deg.append(predictions_LLA_deg)
        self.forecasted_states_LLA_deg_mean.append(predictions_LLA_deg.mean(axis=0))
        self.forecasted_states_LLA_deg_std.append(predictions_LLA_deg.std(axis=0))

        self.forecasted_times.append(crash_times)
        self.forecasted_times_mean.append(crash_times.mean(axis=0))
        self.forecasted_times_std.append(crash_times.std(axis=0))

        if verbose is True:
            print(f'Forecasted mean crash state:\n {self.forecasted_states_mean[-1]}\n')
            print(f'Forecasted standard deviation of crash state:\n {self.forecasted_states_std[-1]}\n')

            print(f'Forecasted mean crash time:\n {self.forecasted_times_mean[-1]}\n')
            print(f'Forecasted standard deviation of crash times:\n {self.forecasted_times_std[-1]}\n')

    def get_outputs(self) -> dict[np.array]:
        """
        This function outputs results from the predictor once it has terminated.

            Inputs:
                    self
            
            Outputs:
                    output: a dictionary of numpy arrays containing various outputs
                            accumulated in the lifespan of the predictor
        """
        
        # Obtain trajectories and times
        prior_trajectories = np.asarray(self.prior_traj)
        posterior_trajectories = np.asarray(self.posterior_traj)
        posterior_trajectories_LLA = np.asarray(self.posterior_traj_states_LLA)
        posterior_trajectories_cartesian = np.asarray(self.posterior_traj_states_cartesian)
        posterior_traj_times = np.asarray(self.posterior_traj_times)

        # Obtain crash site forecast information, such as distributions, averages and standard deviations per MC step in ECF frame
        crash_site_forecasts = np.asarray(self.forecasted_states)
        mean_crash_site_forecasts = np.asarray(self.forecasted_states_mean)
        std_crash_site_forecasts = np.asarray(self.forecasted_states_std)

        # Obtain crash site forecast information, such as distributions, averages and standard deviations per MC step in ECF frame
        crash_site_forecasts_LLA_deg = np.asarray(self.forecasted_states_LLA_deg)
        mean_crash_site_forecasts_LLA_deg = np.asarray(self.forecasted_states_LLA_deg_mean)
        std_crash_site_forecasts_LLA_deg = np.asarray(self.forecasted_states_LLA_deg_std)

        # Obtain crash time forecast information, such as distributions, averages and standard deviations per MC step
        crash_site_times = np.asarray(self.forecasted_times)
        mean_crash_site_times = np.asarray(self.forecasted_times_mean)
        std_crash_site_times = np.asarray(self.forecasted_times_std)

        output_dict = {
            'prior_traj': prior_trajectories,
            'posterior_traj': posterior_trajectories,
            'posterior_traj_LLA': posterior_trajectories_LLA,
            'posterior_traj_cart': posterior_trajectories_cartesian,
            'posterior_traj_times': posterior_traj_times,

            'crash_site_forecasts': crash_site_forecasts,
            'mean_crash_sites': mean_crash_site_forecasts,
            'std_crash_sites': std_crash_site_forecasts,

            'crash_site_forecasts_LLA_degree': crash_site_forecasts_LLA_deg,
            'mean_crash_site_forecasts_LLA_degree': mean_crash_site_forecasts_LLA_deg,
            'std_crash_site_forecasts_LLA_degree': std_crash_site_forecasts_LLA_deg,

            'crash_site_times': crash_site_times,
            'mean_crash_times': mean_crash_site_times,
            'std_crash_times': std_crash_site_times 
        }

        return output_dict
    

def run_predictor(
    predictor, radar_measurements: np.array, active_radar_longitudes: np.array,
    num_samples_MC: int, forecast_gap: int, verbose: bool = True) -> dict:
    """
    This function serves as a wrapper of running the predictor until termination,
    obtaining information about it's estimated trajectories, impact distribution
    and statistics, etc. This is a user-friendly alternative to manually creating
    a loop for the predictor.

        Inputs:
                predictor: an object of the Predictor class
                radar_measurements: the noisy radar station measurements from the simulator
                active_radar_longitudes: the longitudes of the radar stations which provided
                                         the measurements - also given by the simulator
                num_samples_MC: the number of MC samples per forecast step
                forecast_gap: the number of measurements between making forecasts
                verbose: a boolean value to output print statements from the predictor

        Outputs:
                output: a dictionary of results from the predictor such as estimated
                        trajectories, distributions of impact sites and times etc.
    """
    
    # For each measurement recieved from the simulator, run the predictor
    for count, (meas, theta_R) in enumerate(zip(radar_measurements, active_radar_longitudes)):
        
        # obtain the prior state of the satellite at the current time
        predictor.process_model(include_noise=True, verbose=verbose)

        # Evaluate the Jacobian of the process model
        predictor.eval_JacobianF(
            G=G, M_e=M_e, Cd=C_d,
            A=A, m=m_s, R_air=R_air,
            g0=g0, omega_E=omega_E, R_e=R_e, h_s=h_s, verbose=verbose)

        # Update the state covariance matrix to obtain a prior estimate
        predictor.update_prior_belief(verbose=verbose)

        # Obtain the residual/innovation
        predictor.residual(measurement=meas, theta_R=theta_R, verbose=verbose)

        # Evaluate the Jacobian of the measurement model
        predictor.eval_JacobianH(theta_R=theta_R, R_e=R_e, omega_E=omega_E)

        # Compute the Kalman gain matrix
        predictor.kalman_gain(verbose=verbose)

        # Update posterior estimates for the state vector and covariance matrix
        predictor.assimilated_posterior_prediction(verbose=verbose)

        # Forecast every forecast_gap measurements recieved
        if count % forecast_gap == 0 and count > 0:

            predictor.forecast(n_samples=num_samples_MC, final_time=4e9, verbose=verbose)

            # If the longitude forecast is 2 standard deviations less than the required threshold, terminate the predictor 
            if 2*predictor.forecasted_states_std[-1][1] <= predictor_termination:

                print('Two standard deviations of forecasted crash state below 4.7km; terminating predictor.')
                print(f'Predictor terminated after time {predictor.t} seconds.')

                outputs = predictor.get_outputs()
                
                return outputs

        else:
            continue