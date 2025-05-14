import time

import SkyFall
import SkyFall.predictor
import SkyFall.simulator
import SkyFall.simulator.simulator
import SkyFall.utils
from SkyFall.utils.global_variables import *

# Set the delay in which data is fed into the predictor
period: float = 1
n_samples: int = 10
nth_measurement: int = 5

# Initialise the different covariance matrices
P = SkyFall.utils.predictor_utilities.covariance_matrix_initialiser(variances=[1000, 1e-2, 1e-2, 1e-5])
R = SkyFall.utils.predictor_utilities.covariance_matrix_initialiser(variances=[0.1**2, (0.0017)**2])
Q = SkyFall.utils.predictor_utilities.covariance_matrix_initialiser(variances=[1e-3, 1e-6, 1e-3, 1e-6])

# Initialise the initial conditions and timestep

# NB: The state is now in polar coordinates of the form r, theta, r_dot and th_dot
th_dot0 = (np.sqrt(G*M_e / (R_e + 200e3))) / (R_e + 200e3)
x0 = np.array([R_e+200e3, 0.0, 0.0, th_dot0])
del_t = 50
t0 = 0

# Obtain times, real and noisy data from the simulator
sim = SkyFall.simulator.simulator.Simulator(initial_state=x0, measurement_covariance=R, timestep=del_t, t0=t0)
times, real_data, noisy_data, active_radar_indices, active_radar_longitudes, crash_site = sim.get_measurements()

# print(f'Noisy data shape: {noisy_data.shape}')
# print(f'Real data shape: {real_data.shape}')
# print(f'Times shape: {times.shape}')
# print(f'Active radar indices shape: {active_radar_indices.shape}')
# print(f'Active radar longitudes shape: {active_radar_longitudes.shape}')

# Create an object of the predictor
pred = SkyFall.predictor.predictor.Predictor(
    process_covariance=Q,
    measurement_covariance=R,
    state_covariance=P,
    initial_state=x0,
    timestep=del_t,
    t0=t0
)


for count, (meas, theta_R) in enumerate(zip(noisy_data, active_radar_longitudes)):

#     time.sleep(period)

    pred.process_model(include_noise=True, verbose=False)

    pred.eval_JacobianF(
        G=G, M_e=M_e, Cd=C_d,
        A=A, m=m_s, R_air=R_air,
        g0=g0, omega_E=omega_E, R_e=R_e)

    pred.update_prior_belief(verbose=False)

    # This step checks whether or not measurements were recieved or not (in that case, they are NaN)

    # Only update if a measurement is avaliable
    if np.isnan(meas).any() == False or np.isinf(meas).any() == False:

        pred.residual(measurement=meas, theta_R=theta_R, verbose=False)

        pred.eval_JacobianH(theta_R=theta_R, R_e=R_e, omega_E=omega_E)

        pred.kalman_gain(verbose=False)

        pred.assimilated_posterior_prediction(verbose=False)
    
    # Else, no update is made: preceed to the next time step using only the prior state
    else:
        continue

    if count % nth_measurement == 0 and count > 0:

        pred.forecast(n_samples=n_samples, final_time=4e9, verbose=True)

        if 2*pred.forecasted_states_std[-1][0] <= 4700:
            print('Two standard deviations of forecasted crash state below 4.7km; terminating predictor.')
            print(f'Predictor terminated after time {pred.t} seconds.')
            print(f'Final mean forecasted crash site:\nx = {pred.forecasted_states_mean[-1][0]/1000} km\n')
            print(f'True crash site:\n x = {SkyFall.utils.physical_quantities(state=crash_site, initial_state=pred.initial_state)[0]/1000} km\n')
            print(f'Final forecasted state standard deviation:\nx_std = {pred.forecasted_states_std[-1][0]/1000} km\n')
            print(f'Time difference between confident forecast and real data crash: {times[-1] - pred.t} seconds.')
            break

    else:
        continue
    
    break

# # import matplotlib.pyplot as plt

# # plt.figure()
# # plt.plot(np.asarray(pred.posterior_traj_states)[:,0]/1000, np.asarray(pred.posterior_traj_states)[:,1]/1000, marker='s', label='EKF state')
# # plt.plot(real_data[:,0]/1000, real_data[:,1]/1000, marker='x', label='Real measurement data')
# # plt.plot(noisy_data[:,0]/1000, noisy_data[:,1]/1000, marker='.', label='Noisy measurement data ')
# # plt.xlabel('Distance travelled (km)')
# # plt.ylabel('Altitude (km)')
# # plt.legend(loc='best')
# # plt.show()


# # # fig, ax = plt.subplots(1, 2)
# # # ax[0].hist(pred.forecasted_states[0][:,0]/1000, label='First forecast')
# # # ax[0].set_xlabel('Distance travelled (km)')
# # # ax[0].legend()

# # # ax[1].hist(pred.forecasted_states[-1][:,0]/1000, label='Final forecast')
# # # ax[1].set_xlabel('Distance travelled (km)')
# # # ax[1].legend()

# # # plt.show()

