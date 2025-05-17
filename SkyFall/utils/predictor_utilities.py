"""
This file defines some utility functions that are required for the predictor and simulator.

Module: ES98B
Group: ISEE-3
"""

from scipy.integrate import solve_ivp

# Load in constant variables (e.g. Earth's radius)
from .global_variables import *

def covariance_matrix_initialiser(variances: np.array, covariances=None) -> np.array:
    """
    This function intialises a covariance matrix for provided variances and covariances
    arrays; this can be used to compute covariance matrices or a single array containing
    a variance value for a single variable for any process (e.g. process model, state,
    measurement etc.).

        Inputs:
                variances: an array of the variance(s)
                covariances: an array of the covariances for multiple variables

        Output:
                cov_mat: a valid (symmetric, square) covariance matrix 
    """

    assert np.isfinite(variances).all() == True, "Variances must be finite or non-NaN values"

    if covariances is not None:
        assert np.isfinite(covariances).all() == True, "Covariances must be finite or non-NaN values"
    
    # Ensures the passed in arguements are Numpy arrays (not lists) 
    variances = np.asarray(variances)

    # If covariance is provided, we intialise a non-diagonal covariance matrix
    if variances.size > 1 and covariances is not None:

        covariances = np.asarray(covariances)
        
        # Obtain matrix of covariance matrix
        matrix_dim: int = variances.shape[0]

        # Create matrix and populate
        cov_mat: np.array = np.zeros(shape=(matrix_dim, matrix_dim))

        # Obtain the indices of the upper triangle (excluding diagonal)
        upper_diag_indices = np.triu_indices(matrix_dim, k=1)

        # Populate the off-diagonals
        for enum, (i_index, j_index)  in enumerate(zip(upper_diag_indices[0], upper_diag_indices[1])):
            cov_mat[i_index][j_index] = covariances[enum]

        # Populate the lower triangle too (due to symmetry)
        cov_mat = cov_mat + cov_mat.T

        # Populate the diagonal
        for i_index in range(matrix_dim):
            cov_mat[i_index][i_index] = variances[i_index]

    # Else, the covariance matrix is diagonal (i.e. independent variables)
    elif variances.size > 1 and covariances is None:
        cov_mat = np.diag(v=variances)
    
    # Else, we are working with a single random variables and return 1D array of just the variance
    else:
        # reshape(1, 1) ensures a shape consistent with what is required for the linear algebra in other functions 
        cov_mat = variances.reshape(1, 1)

    return cov_mat

# def barometric_air_density(y: float) -> float:
#     """
#     This function returns the air density at a given
#     altitude y using the Barometric formula.

#         Input:
#                 y: Altitude of satellite (m) above Earth's surface
        
#         Output:
#                 rho: The air density at altitude y
#     """

#     # Pick highest b with h_b <= y
#     hb   = layers[0]["h"]
#     rhob = layers[0]["rho"]
#     Tb   = layers[0]["T"]

#     for b in reversed(range(len(layers))):
#         if y >= layers[b]["h"]:
#             hb   = layers[b]["h"]
#             rhob = layers[b]["rho"]
#             Tb   = layers[b]["T"]
#             break
    
#     # Compute air density as a function of altitude (Barometric formula) 
#     rho = rhob * np.exp(-g0 * M_molar * (y - hb) / (R_star * Tb))
    
#     return rho

def USA76_air_density(y: float) -> float:
    """
    This function returns the air density at a given
    altitude y based on the U.S. Standard Atmosphere
    1976 model.

        Input:
                y: Altitude of satellite (m)
        
        Output:
                rho: The air density at altitude y
    """
    if y < 0:
        y = 0.0
    
    if y > 86e3:                      
        h_s  = 7000.0
        rho = base_rho[-1]*np.exp(-(y-86e3)/h_s)
        return rho
    
    for index, (h_b, _, _) in enumerate(layers):
        if y >= h_b:
            layer = index

    h_b, T_b, L_b = layers[layer]
    h    = y - h_b
    rho_b = base_rho[layer]

    if L_b != 0.0:
        T   = T_b + L_b*h
        rho = rho_b * (T_b/T)**(1 + g0/(R_air*L_b))

    else:
        rho = rho_b * np.exp(-g0*h/(R_air*T_b))

    return rho


def equations_of_motion(time: float, state: np.array) -> list[float]:
    """
    This function contains the process model (dynamics of the satellite)
    ; the equations of motion (EoMs) in terms of the global coordinates
    about Earth's center.

        Inputs:
                time: The current timestep
                state: The current state of the satellite (positions and velocities in x-y directions)

        Outputs:
                f: Return the process model
    """

    assert time >= 0, "Time value must be a non-negative float"
    assert np.isfinite(state).all() == True, "State values must be finite or non-NaN values"

    # Extract components of the state and compute air density 
    r, theta, r_dot, th_dot = state

    # Used this to prevent any exp(+inf) overflows
    altitude = max(r - R_e, 0.0)
    rho = USA76_air_density(y=altitude)

    # Compute relative speed to rotating atmosphere
    v_rel = np.hypot(r_dot, r*(th_dot - omega_E))

    # Test to see if v_rel is small to prevent any numerical errors
    if abs(v_rel) <= 1e-8:
        raise ValueError('Magnitude of relative velocity is too small - risk of numerical errors')
    
    # Compute intermediary variables
    drag = (0.5 * rho * C_d * A) / (m_s)
    g_centre = (G*M_e) / (r**2)

    a_r_drag = -drag * v_rel * r_dot
    a_th_drag = -drag * v_rel * (r*(th_dot - omega_E))

    # Compute acceleration components
    r_dotdot = (r*th_dot**2 - g_centre + a_r_drag)
    th_dotdot = ((-2 * r_dot * th_dot) + a_th_drag) / (r)

    # Output the process model
    f = [r_dot, th_dot, r_dotdot, th_dotdot]

    return f

def hit_ground(time: float, state: np.array) -> float:
    """
    This function defines a termination criteria for the
    ODE solver; terminate when the satellite makes contact with
    the surface of the Earth.
    """

    assert time >= 0, "Time value must be a non-negative float"
    assert np.isfinite(state).all() == True, "State values must be finite or non-NaN values"

    R_e = 6.371e6

    return state[0] - R_e
hit_ground.terminal = True
hit_ground.direction = -1  # only triggered when y decreasing


def longitude_cal(distance: float) -> float:
    """
    This function converts the distance travelled by the
    satellite in the x-direction (m) into longitude.

    NB: This treats distance to be calculated by the satellite
        travelling eastward from the Null island i.e. the place
        with (0,0) longitude and latitude.

        Inputs:
                distance: The distance travelled by the satellite
                          in the x-direction (m)
        
        Outputs:
                longitude: The longitude at given distance 
    """

    assert distance > 0, "Distance must be a non-negative float"

    one_degree_longitude_to_km = 40075 / 360

    degrees = distance / one_degree_longitude_to_km
    longitude = (degrees + 180)%360 - 180

    return longitude

def polar_to_cartesian_state(state: np.array) -> np.array:
    """
    This function converts the state vector, which is in polar
    coordinates, into a state vector in the Cartesian coordinate
    system.

        Inputs:
                state: a state vector of the satellite

        Outputs:
                cartesian_state: a state vector in the Cartesian
                                 system
    """

    assert np.isfinite(state).all() == True, "State values must be finite or non-NaN values"

    r, theta, r_dot, th_dot = state

    # Convert into the Cartesian system
    x = r * np.cos(theta)
    x_dot = r_dot * np.cos(theta) - r * th_dot * np.sin(theta)
    y = r * np.sin(theta)
    y_dot = r_dot * np.sin(theta) + r * th_dot * np.cos(theta)

    cartesian_state = np.array([x, y, x_dot, y_dot])

    return cartesian_state


def measurement_model_h(state: np.array, radar_longitude: float) -> np.array:
    """
    This function implements the measurement model h, which converts the prior state
    of the predictor (EKF) at time t and converts it into a measurement expected
    to be recieved from the radar station at time t.

        Inputs:
                state: the prior state of the predictor at time t (x_bar)
                radar_longitude: the longitude of the 'active radar' station at time t
                                 which provided the measurement to the predictor in radians

        Outputs:
                h: the measurement model output h(state)
    """

    assert np.isfinite(state).all() == True, "State values must be finite or non-NaN values"

    # Extract state components
    r, theta, r_dot, th_dot = state

    # Compute the position of the satellite in Cartesian coordinates
    r_s = np.array([r*np.cos(theta), r*np.sin(theta)])

    # Compute the position of the 'active' radar station at time t in Cartesian coordinates
    r_r = np.array([R_e*np.cos(radar_longitude), R_e*np.sin(radar_longitude)])

    # Compute the velocity of the satellite in Cartesian coordinates
    v_s = polar_to_cartesian_state(state)[-2:]

    # Compute the velocity of the radar station in Cartesian coordinates
    v_r = np.array([-R_e*omega_E*np.sin(radar_longitude), R_e*omega_E*np.cos(radar_longitude)])

    # Compute the radial distance (range) between the satellite and the radar station
    eta = np.linalg.norm(x=(r_s - r_r))

    # Compute the radial velocity between the satellite and the radar station
    eta_dot = ((v_s - v_r).T @ (r_s - r_r)) / (eta)

    # Return array for the radar measurement
    h = np.array([eta, eta_dot])

    return h


def physical_quantities(state: np.array, initial_state: np.array) -> np.array:
    """
    This function converts the polar states into physical quantities useful for
    the visualiser, such as the distance travelled along the equator from the
    starting point and the altitude of the satellite at time t.

        Inputs:
                state: The state of the satellite at the current time
                initial_state: The initial state of the satellite, x0

        Outputs:
                physical_state: Returns the physical quantities of the state
    """

    # Extract the components of the current state
    r, theta, r_dot, th_dot = state

    # Compute distance travelled along the equator from the starting point
    # Make it into [0, 2pi]
    theta = theta % (2 * np.pi)
    distance_travelled = R_e * ((theta - initial_state[1]) % (2 * np.pi))

    # Compute altitude
    altitude = r - R_e
    
    # Compute velocities
    v_x = polar_to_cartesian_state(state=state)[-2]
    v_y = polar_to_cartesian_state(state=state)[-1]

    physical_state = np.array([distance_travelled, altitude, v_x, v_y])

    return physical_state

def ECI_to_ECEF(time: float, state: np.array) -> tuple[np.array, np.array, np.array]:
    """
    This function converts posterior and forecasted states from ECI to ECEF - which
    is essential for visualisation.

        Inputs:
                time: current time of the state
                state: forecasted or posterior state at time

        Outputs:
                state_ECEF: the state in polar coordinates in ECEF frame
                state_ECEF_cartesian: the state in cartesian coordinates in ECEF frame
                LLA: the latitude, longitude and altitude of state
    """

    state = np.asarray(state)

    # Extract state components
    r, theta, r_dot, th_dot = state

    # Convert from ECI to ECEF
    theta_ECEF = theta - omega_E * time
    th_dot_ECEF = th_dot - omega_E

    state_ECEF = np.array([r, theta_ECEF, r_dot, th_dot_ECEF])

    # Convert into Cartesian coordinates
    state_ECEF_cartesian = polar_to_cartesian_state(state=state_ECEF)

    # Obtain latitude, longitude and altitude
    lat = 0.0   # Always 0 due to equitorial orbit
    lon = np.arctan2(state_ECEF_cartesian[1], state_ECEF_cartesian[0])
    altitude = r - R_e

    LLA_radians = np.array([lat, lon % (2 * np.pi), altitude])
    LLA_deg = np.array([np.degrees(lat), np.degrees(lon), altitude])

    return state_ECEF, state_ECEF_cartesian, LLA_radians, LLA_deg