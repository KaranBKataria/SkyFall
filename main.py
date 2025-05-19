"""
This script serves as an example for the user to understand how to use
SkyFall, what user-defined arguements are required, the form of the output,
what can be tweaked etc.

Module: ES98B
Group: ISEE-3
"""

from SkyFall.predictor import Predictor, run_predictor
from SkyFall.simulator.simulator import Simulator
from SkyFall.visualiser.visualiser import Visualiser
from SkyFall.utils.global_variables import *
from SkyFall.utils import predictor_utilities

import matplotlib.pyplot as plt

def main():

    # Define user-specified parameters
    
    # Number of forecast samples 
    n_samples: int = 5

    # Number of measurements between forecasts made
    nth_measurement: int = 5

    # Define the covariance matrices (NB: no covariances have been included, but this can optionally be added)

    # State covariance matrix (4x4)
    P: np.array = predictor_utilities.covariance_matrix_initialiser(variances=[3000, 1, 1000, 0.5], covariances=None)

    # Measurement covariance matrix (2x2)
    R: np.array = predictor_utilities.covariance_matrix_initialiser(variances=[1000**2, 4], covariances=None)

    # # Process covariance matrix (4x4)
    Q: np.array = predictor_utilities.covariance_matrix_initialiser(variances=[20**2, 0.0001**2, 0.2**2, 1e-14], covariances=None)

    # Initialise the initial conditions and timestep

    # NB: The state is in polar coordinates of the form r, theta, r_dot and th_dot
    # Compute alitutude (m) and hence distance from Earth's centre
    h: float = 150e3
    r: float = R_e + h

    v_c = np.sqrt(G*M_e / r)

    # Compute initial angular velocity (> 0 for prograde motion; mimicing realistic satellite orbits)
    th_dot0 = (v_c - 10)/r

    # Initial state must be an array of shape 4x1
    x0: np.array = np.array([r, 0.0, 0.0, th_dot0])

    # Define the initial time and time step
    del_t: float = 20
    t0: float = 0.0

    # Obtain times, real and noisy data from the simulator
    sim = Simulator(initial_state=x0, measurement_covariance=R, timestep=del_t, t0=t0)
    times, real_data, noisy_data, active_radar_indices, active_radar_longitudes, crash_site, crash_time, real_traj \
        = sim.get_measurements()
        
    print(crash_site)

    # Create an object of the predictor
    pred = Predictor(
        process_covariance=Q,
        measurement_covariance=R,
        state_covariance=P,
        initial_state=x0,
        timestep=del_t,
        t0=t0
    )

    # For each measurement recieved, run the predictor
    for count, (meas, theta_R, traj) in enumerate(zip(real_data, active_radar_longitudes, real_traj)):
        
        print(f'Simulator ODE output {count+1}: {traj}')

        print(f'Process alt {pred.posterior_state[0] - R_e}')
        pred.process_model(include_noise=True, verbose=True)
        print(f'Jacobian alt {pred.prior_state[0] - R_e}')
        pred.eval_JacobianF(
            G=G, M_e=M_e, Cd=C_d,
            A=A, m=m_s, R_air=R_air,
            g0=g0, omega_E=omega_E, R_e=R_e, h_s=h_s, verbose=True)

        pred.update_prior_belief(verbose=False)

        pred.residual(measurement=meas, theta_R=theta_R, verbose=False)

        pred.eval_JacobianH(theta_R=theta_R, R_e=R_e, omega_E=omega_E)

        pred.kalman_gain(verbose=False)

        pred.assimilated_posterior_prediction(verbose=True)

        if count % nth_measurement == 0 and count > 0:

            pred.forecast(n_samples=n_samples, final_time=4e9, verbose=True)

            # If the longitude forecast is 2 standard deviations less than the required threshold, terminate the predictor 
            if 2*pred.forecasted_states_std[-1][1] <= predictor_termination:

                print('Two standard deviations of forecasted crash state below 4.7km; terminating predictor.')
                print(f'Predictor terminated after time {pred.t} seconds.')

                outputs = pred.get_outputs()
                posterior_trajectories_LLA = outputs['posterior_traj_LLA']
                prior_traj = outputs['prior_traj']
                posterior_traj = outputs['posterior_traj']
                traj_times = outputs['posterior_traj_times']
                posterior_trajectories_cartesian = outputs['posterior_traj_cart']

                forecasted_crash_LLA = outputs['crash_site_forecasts']
                # forecasted_crash_LLA = outputs['crash_site_forecasts_LLA_degree']
                print(f'Shape of forecasted_crash_LLA: {forecasted_crash_LLA.shape}')
                mean_forecasted_crash_LLA = outputs['mean_crash_site_forecasts_LLA_degree']
                mean_forecasted_crash_time = outputs['mean_crash_times']

                print(f'Mean time of forecasted crash from t0 = {t0} seconds: {mean_forecasted_crash_time[-1][0]} seconds.')
                print(f'Actual time of crash from t0 = {t0} seconds: {crash_time} seconds.')
                
                vis = Visualiser(
                    times=traj_times,
                    trajectory_cartesian=posterior_trajectories_cartesian,
                    trajectory_LLA=posterior_trajectories_LLA,
                    crash_lon_list=forecasted_crash_LLA
                )

                vis.plot_orbit()

                vis.animate_orbit()

                vis.plot_crash_distribution()

                vis.plot_height_vs_time()

                vis.plot_orbit_map()

                break

        else:
            continue
        

    plt.figure()
    plt.plot(traj_times, posterior_trajectories_LLA[:,-1]/1000, marker='s', label='Posterior EKF state')
    # plt.plot(traj_times, prior_traj[:,-1]/1000, marker='s', label='EKF state')
    plt.plot(times, (real_traj[:,0]-R_e)/1000, marker='x', label='Real measurement data')
    # plt.xlim(0, 1000)
    # plt.plot(noisy_data[:,0]/1000, noisy_data[:,1]/1000, marker='.', label='Noisy measurement data ')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (km)')
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    main()
