"""
This script serves as an example for the user to understand how to use
SkyFall, what user-defined arguements are required, the form of the output,
what can be tweaked etc.

Module: ES98B
Group: ISEE-3
"""

## THIS IS A WORKING FILE - PERFECT BEFORE FOR THE FINAL SUBMISSON

from SkyFall.predictor import Predictor
from SkyFall.simulator.simulator import Simulator
from SkyFall.visualiser.visualiser import Visualiser
from SkyFall.utils.global_variables import *
from SkyFall.utils import predictor_utilities

import matplotlib.pyplot as plt
# from filterpy.common import Q_discrete_white_noise # 17th May Changes

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

def main():

    # Define user-specified parameters
    
    # Number of forecast samples 
    n_samples: int = 1

    # Number of measurements between forecasts made
    nth_measurement: int = 10

    # Define predictor termination criteria (by default <= 4700m or equivalently <= 0.0007377 in radians)
    predictor_termination: float = 0.0007377

    # Define the covariance matrices (NB: no covariances have been included, but this can optionally be added)

    # State covariance matrix (4x4)
    P: np.array = predictor_utilities.covariance_matrix_initialiser(variances=[3000, 1, 1000, 0.5], covariances=None)
    # P: np.array = predictor_utilities.covariance_matrix_initialiser(variances=[100, 1e-2, 100, 1e-4], covariances=None)

    # Measurement covariance matrix (2x2)
    # R: np.array = predictor_utilities.covariance_matrix_initialiser(variances=[10**2, (0.0005)**2], covariances=None)
    # R: np.array = predictor_utilities.covariance_matrix_initialiser(variances=[1e8, 1e4], covariances=[1e2])
    R: np.array = predictor_utilities.covariance_matrix_initialiser(variances=[200**2, 4], covariances=None)


    # # Process covariance matrix (4x4)
    # Q: np.array = predictor_utilities.covariance_matrix_initialiser(variances=[10, 1e-4, 1, 1e-8], covariances=[0,2,0,0,1e-6,0])
    Q: np.array = predictor_utilities.covariance_matrix_initialiser(variances=[25, 0.0001**2, 0.1**2, 1e-12], covariances=None)

    # Initialise the initial conditions and timestep

    # NB: The state is in polar coordinates of the form r, theta, r_dot and th_dot
    # Compute alitutude (m) and hence distance from Earth's centre
    h: float = 200e3
    r: float = R_e + h

    v_c = np.sqrt(G*M_e / r)
    print(v_c)

    # Compute initial angular velocity (> 0 for prograde motion; mimicing realistic satellite orbits)
    # th_dot0: float = ((np.sqrt(G*M_e / r) - 30)  / r)
    th_dot0 = (v_c - 30)/r

    # Initial state must be an array of shape 4x1
    x0: np.array = np.array([r, 0.0, 0.0, th_dot0])

    del_t: float = 20
    t0: float = 0.0

    # Obtain times, real and noisy data from the simulator
    sim = Simulator(initial_state=x0, measurement_covariance=R, timestep=del_t, t0=t0)
    times, real_data, noisy_data, active_radar_indices, active_radar_longitudes, crash_site, crash_time, real_traj \
        = sim.get_measurements()
        
    print(crash_site)

    # print(f'Noisy data shape: {noisy_data.shape}')
    # print(f'Real data shape: {real_data.shape}')
    # print(f'Times shape: {times.shape}')
    # print(f'Active radar indices shape: {active_radar_indices.shape}')
    # print(f'Active radar longitudes shape: {active_radar_longitudes.shape}')

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

        # print(f'Theta_R: {theta_R}')

        pred.process_model(include_noise=True, verbose=False)

        pred.eval_JacobianF(
            G=G, M_e=M_e, Cd=C_d,
            A=A, m=m_s, R_air=R_air,
            g0=g0, omega_E=omega_E, R_e=R_e, verbose=True)

        pred.update_prior_belief(verbose=False)

        # This step checks whether or not measurements were recieved or not (in that case, they are NaN)

        # Only update if a measurement is avaliable
        # if np.isnan(meas).any() == False or np.isinf(meas).any() == False:

        pred.residual(measurement=meas, theta_R=theta_R, verbose=False)

        pred.eval_JacobianH(theta_R=theta_R, R_e=R_e, omega_E=omega_E)

        pred.kalman_gain(verbose=False)

        pred.assimilated_posterior_prediction(verbose=True)
        
        # Else, no update is made: preceed to the next time step using only the prior state
        # else:
        #     continue

        if count % nth_measurement == 0 and count > 0:

            pred.forecast(n_samples=n_samples, final_time=4e9, verbose=True)

            # If the longitude forecast is 2 standard deviations less than the required threshold, terminate the predictor 
            if 2*pred.forecasted_states_std[-1][1] <= predictor_termination:

                print('Two standard deviations of forecasted crash state below 4.7km; terminating predictor.')
                print(f'Predictor terminated after time {pred.t} seconds.')

                outputs = pred.get_outputs()
                posterior_trajectories_LLA = outputs['posterior_traj_LLA']
                print(f'Shape: {posterior_trajectories_LLA.shape}')
                print(f'LLA Trajectories: {posterior_trajectories_LLA}')

                prior_traj = outputs['prior_traj']
                posterior_traj = outputs['posterior_traj']
                traj_times = outputs['posterior_traj_times']

                posterior_trajectories_cartesian = outputs['posterior_traj_cart']

                forecasted_crash_LLA = outputs['crash_site_forecasts']
                mean_forecasted_crash_LLA = outputs['mean_crash_sites']
                mean_forecasted_crash_time = outputs['mean_crash_times']

                # print(f'Final mean forecasted crash site:\n Latitude: 0 deg, Longitude: {mean_forecasted_crash_LLA[-1][]}\n')
                print(f'Mean time of forecasted crash from t0 = {t0} seconds: {mean_forecasted_crash_time[-1][0]} seconds.')
                print(f'Actual time of crash from t0 = {t0} seconds: {crash_time} seconds.')
                # print(f'True crash site:\n x = {predictor_utilities.physical_quantities(state=crash_site, initial_state=pred.initial_state)[0]/1000} km\n')
                # print(f'Final forecasted state standard deviation:\nx_std = {pred.forecasted_states_std[-1][0]/1000} km\n')
                # print(f'Time difference between confident forecast and real data crash: {times[-1] - pred.t} seconds.')
                
                vis = Visualiser(
                    times=traj_times,
                    trajectory_cartesian=posterior_trajectories_cartesian,
                    trajectory_LLA=posterior_trajectories_LLA,
                    crash_lon_list=forecasted_crash_LLA
                )

                # vis.plot_orbit()

                # vis.animate_orbit()

                vis.plot_crash_distribution()

                vis.plot_height_vs_time()

                vis.plot_orbit_map()

                break

        else:
            continue
        

    plt.figure()
    plt.plot(traj_times, posterior_trajectories_LLA[:,-1]/1000, marker='s', label='EKF state')
    plt.plot(traj_times, (prior_traj[:,0] - R_e)/1000, linestyle='--', label='Prior EKF state')
    plt.plot(times, (real_traj[:,0]-R_e)/1000, marker='x', label='Real measurement data')
    # plt.xlim(0, 1000)
    # plt.plot(noisy_data[:,0]/1000, noisy_data[:,1]/1000, marker='.', label='Noisy measurement data ')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (km)')
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":

    main()
