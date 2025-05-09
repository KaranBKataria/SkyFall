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

        # Track the posterior trajectory
        self.posterior_traj_states = [initial_state]
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

        self.forecasted_times = []
        self.forecasted_times_mean = []
        self.forecasted_times_std = []

    def process_model(self, include_noise=False, verbose: bool = True):# -> np.array:
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
        #global t0
        t_eval = np.arange(t, t + 2*timestep, timestep)

        # Numerically solve the ODE up to the next time step
        predicted_state = solve_ivp(
            fun=equations_of_motion,
            t_span=[t, t + timestep],
            y0=state,
            method='RK23',
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
            predicted_state_with_noise = predicted_state.y[:,-1] + additive_gaussian_noise

            self.prior_state = predicted_state_with_noise 

            #return predicted_state_with_noise
        
        else:
            predicted_state_without_noise = predicted_state.y[:,-1]
            
            self.prior_state = predicted_state_without_noise
            #return predicted_state_without_noise    
        
        if verbose is True:
            print(f'Current prior state:\n {self.prior_state}\n')
    
    def measurement_model(self):
        return self.prior_state

    def eval_JacobianF(self, G=G, M_e=M_e, Cd=C_d, A=A, m=m_s, R_star=R_star, g0=g0, M_molar=M_molar) -> np.array:

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

            Outputs:
                    self.JacobianF: The Jacobian evaluated at the posterior state of the
                                    previous time step
        """
    
        # Ensure posterior state of previous time step is an array and extract it's components
        state = np.asarray(self.posterior_state)
        x, y, vx, vy = state

        # Select correct parameters of the Barometric formula based on altitude (y)
        for b in reversed(range(len(layers))):
            if y >= layers[b]["h"]:
                h_b   = layers[b]["h"]
                rho_b = layers[b]["rho"]
                T_b   = layers[b]["T"]
                break

        # Evaluate the Jacobian F
        F = F_func(x, y, vx, vy, G, M_e, Cd, A, m, rho_b, R_star, g0, T_b, h_b, M_molar)

        self.JacobianF = F

    def eval_JacobianH(self):
        
        n_dims = np.asarray(self.prior_state).size
        H = np.eye(N=n_dims)
        self.JacobianH = H

    def update_prior_belief(
        self, JacobianV=None, control_covariance=None, verbose: bool = True):# -> np.array:
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
            self.prior_state_covariance = P_bar

            #return P_bar
        
        # Else, update the state covariance without any control inputs
        else: 
            P_bar = (F @ P @ F.T) + Q 
            self.prior_state_covariance = P_bar

            #return P_bar

        if verbose is True:
            print(f'Current prior state covariance matrix:\n {self.prior_state_covariance}\n')

    def residual(self, measurement: np.array, measurement_model=measurement_model, verbose: bool = True):# -> np.array:
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
        residual = np.array(measurement - measurement_model(self))

        # If there is a single value (i.e. size-1 array), reshape it for compatibility with later linear algebra
        if residual.size == 1:
            residual = residual.reshape(1,1)

        self.res = residual
        #return res

        if verbose is True:
            print(f'Residual value:\n {self.res}\n')

    def kalman_gain(self, verbose: bool = True) -> np.array:
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
            P = P.reshape(1, 1)
        elif R.ndim == 1:
            R = R.reshape(1, 1)
        elif H.ndim == 1:
            H = H.reshape(1, 1)
                
        # Create intermediate variable
        S = H @ P_bar @ H.T + R
    
        # Compute the Kalman Gain using linear solve (not inversion due to numerical instability)
        K = np.linalg.solve(S, (P_bar @ H.T).T).T

        self.kalman_gain_matrix = K
        #return K

        if verbose is True:
            print(f'Current Kalman gain:\n {self.kalman_gain_matrix}\n')

    def assimilated_posterior_prediction(self, verbose: bool = True):# -> np.array, np.array:
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
        self.posterior_state_covariance = P_assimilated

        # Compute the posterior (data assimilated) state vector
        x_state_assimilated = x_bar + (K @ y)
        self.posterior_state = x_state_assimilated

        if verbose is True:
            print(f'Current posterior state:\n {self.posterior_state}\n')
            print(f'Current posterior state covariance matrix:\n {self.posterior_state_covariance}\n')

        # Append to posterior trajectories list
        self.posterior_traj_states.append(self.posterior_state)
        self.posterior_traj_times.append(self.t)
        #return x_state_assimilated, P_assimilated

    def forecast(self, n_samples: int, final_time=20_000, verbose: bool = True):#-> (np.array, np.array):
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

        #  Extract CURRENT time and timestep
        t = self.t
        timestep = self.timestep

        # Define lists to be populated with sample crash predictions and timings
        predictions = []
        crash_times = []

        # Draw samples from the state distribution, centered on the state with a state covariance matrix
        samples = np.random.multivariate_normal(mean=state, cov=state_covariance, size=n_samples)
    
        t_eval = np.arange(t, final_time + timestep, timestep)

        # For each sample, solve the ODE system to obtain a distribution - this is sampling-based uncertainty propogation
        for sample in samples:
            predicted_state = solve_ivp(
                fun=equations_of_motion,
                t_span=[t, final_time],
                y0=sample,
                method='RK23',
                t_eval=t_eval,
                events=hit_ground,
                rtol=1e-8,
                atol=1e-8
            )

            predictions.append(predicted_state.y_events)
            crash_times.append(predicted_state.t_events)

        # Return the distribution of predictions and timings, and their statistics, reshaped into an appropriate format
        predictions = np.array(predictions).reshape(n_samples, state.shape[0])
        crash_times = np.array(crash_times).reshape(n_samples, 1)

        self.forecasted_states.append(predictions)
        self.forecasted_states_mean.append(predictions.mean(axis=0))
        self.forecasted_states_std.append(predictions.std(axis=0))

        self.forecasted_times.append(crash_times)
        self.forecasted_times_mean.append(crash_times.mean(axis=0))
        self.forecasted_times_std.append(crash_times.std(axis=0))
        #return np.array(predictions).reshape(n_samples, state.shape[0]), np.array(crash_times).reshape(n_samples, 1)

        if verbose is True:
            print(f'Forecasted mean crash state:\n {np.array(predictions).reshape(n_samples, state.shape[0]).mean(axis=0)}\n')
            print(f'Forecasted standard deviation of crash state:\n {np.array(predictions).reshape(n_samples, state.shape[0]).std(axis=0)}\n\n')

            print(f'Forecasted mean crash time:\n {np.array(crash_times).reshape(n_samples, 1).mean(axis=0)}\n')
            print(f'Forecasted standard deviation of crash times:\n {np.array(crash_times).reshape(n_samples, 1).std(axis=0)}\n')


if __name__ == "__main__":

    # This toy problem tests the functionality of the predictor class;  the values
    # are choosen arbitrarily

    P = covariance_matrix_initialiser(variances=[0.1, 0.5, 0.3, 0.5])
    R = covariance_matrix_initialiser(variances=[0.1, 0.2, 2, 3])
    Q = covariance_matrix_initialiser(variances=[0.9, 0.3, 4, 5])
    # F = np.eye(4)
    # H = np.eye(4)*2
        
    # h = lambda x: x * 0.3

    x0 = np.array([0.0, 120e3, 7.8e3, 0])
    z = np.array([7.79986841e+03, 1.19995270e+05, 7.79973680e+03, -9.46008824e+00])
    del_t = 0.1
    t0 = 0

    pred = Predictor(
        process_covariance=Q,
        measurement_covariance=R,
        state_covariance=P,
        initial_state=x0,
        timestep=del_t,
        t0=t0
    )

    print(f'Intial time t0: {pred.t}')
    print(f'Intial prior state: {pred.prior_state}\n')
    pred.process_model()
    # print(f'Time of first iteration {pred.t}')
    # print(f'Prior state after first iteration {pred.prior_state}\n')

    pred.eval_JacobianF()
    # print(f'Jacobian F:\n {pred.JacobianF}\n')

    # print(f'Intial state covariance:\n {pred.posterior_state_covariance}\n')
    pred.update_prior_belief()
    # print(f'Updated, prior state covariance:\n {pred.prior_state_covariance}\n')

    pred.residual(measurement=z)
    # print(f'Residual value:\n {pred.res}\n')

    pred.eval_JacobianH()
    # print(f'The H matrix is:\n {pred.JacobianH}\n')

    pred.kalman_gain()
    # print(f'Kalman gain:\n {pred.kalman_gain_matrix}\n')

    # print(f'Prior state:\n {pred.prior_state}\n')
    # print(f'Prior state covariance:\n {pred.prior_state_covariance}\n')
    pred.assimilated_posterior_prediction()
    # print(f'Posterior state:\n {pred.posterior_state}\n')
    # print(f'Posterior state covariance:\n {pred.posterior_state_covariance}\n')

    pred.forecast(n_samples=10)