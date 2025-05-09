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


def air_density(y: float) -> float:
    """
    This function returns the air density at a given
    altitude y.

        Input:
                y: Altitude of satellite (m)
        
        Output:
                rho: The air density at altitude y
    """

    # Pick highest b with h_b <= y
    hb   = layers[0]["h"]
    rhob = layers[0]["rho"]
    Tb   = layers[0]["T"]

    for b in reversed(range(len(layers))):
        if y >= layers[b]["h"]:
            hb   = layers[b]["h"]
            rhob = layers[b]["rho"]
            Tb   = layers[b]["T"]
            break
    
    # Compute air density as a function of altitude (Barometric formula) 
    rho = rhob * np.exp(-g0 * M_molar * (y - hb) / (R_star * Tb))
    
    return rho


def equations_of_motion(time: float, state: np.array) -> list[float]:
    """
    This function contains the process model (dynamics of the satellite)
    ; the equations of motion (EoMs).

        Inputs:
                time: The current timestep
                state: The current state of the satellite (positions and velocities in x-y directions)

        Outputs:
                f: Return the process model

    """

    # Extract components of the state and compute air density 
    x, y, vx, vy = state
    rho = air_density(y)

    # Compute relative speed
    v_rel = np.hypot(vx, vy)

    # Test to see if v_rel is small to prevent division by small number
    if abs(v_rel) <= 1e-8:
        raise ValueError('Magnitude of relative velocity is too small - division by small value')
    
    # If vel_rel is too close to 0, skip division
    if v_rel>0:
        a_drag = 0.5 * (rho * C_d * A / m_s) * v_rel**2
        ax_drag = -a_drag * vx / v_rel
        ay_drag = -a_drag * vy / v_rel
    else: 
        ax_drag = ay_drag = 0.0

    # Compute acceleration due to gravity in the y-component
    ay_grav = -G * M_e / (R_e + y)**2

    f = [vx, vy, ax_drag, ay_drag + ay_grav]

    return f

def hit_ground(t: float, state: np.array) -> float:
    """
    This function defines a termination criteria for the
    ODE solver; terminate when the satellite makes contact with
    the surface of the Earth.
    """
    return state[1]

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

    one_degree_longitude_to_km = 40075 / 360

    degrees = distance / one_degree_longitude_to_km
    longitude = (degrees + 180)%360 - 180

    return longitude
