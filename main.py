import time
# import numpy as np

import SkyFall
import SkyFall.predictor
import SkyFall.simulator
import SkyFall.simulator.simulator
import SkyFall.utils
from SkyFall.utils.global_variables import *

# Set the delay in which data is fed into the predictor
period: float = 1

# Initialise the different covariance matrices
P = SkyFall.utils.predictor_utilities.covariance_matrix_initialiser(variances=[10000, 10000, 10000, 10000])
R = SkyFall.utils.predictor_utilities.covariance_matrix_initialiser(variances=[100_000, 100_000, 100_000, 100_000])
Q = SkyFall.utils.predictor_utilities.covariance_matrix_initialiser(variances=[1000, 1000, 1000, 1000])

# Initialise the initial conditions and timestep
x0 = np.array([0.0, 600e3, 100e3, 0])
del_t = 10
t0 = 0

# Obtain times, real and noisy data from the simulator
sim = SkyFall.simulator.simulator.Simulator(initial_state=x0, measurement_covariance=R, timestep=del_t, t0=t0)
times, real_data, noisy_data = sim.get_measurements()

# Create an object of the predictor
pred = SkyFall.predictor.predictor.Predictor(
    process_covariance=Q,
    measurement_covariance=R,
    state_covariance=P,
    initial_state=x0,
    timestep=del_t,
    t0=t0
)


for count, meas in enumerate(noisy_data):

    time.sleep(period)

    pred.process_model(verbose=False)

    pred.eval_JacobianF(
        G=G, M_e=M_e, Cd=C_d,
        A=A, m=m_s, R_star=R_star,
        g0=g0, M_molar=M_molar)

    pred.update_prior_belief(verbose=False)

    pred.residual(measurement=meas, verbose=False)

    pred.eval_JacobianH()

    pred.kalman_gain(verbose=False)

    pred.assimilated_posterior_prediction(verbose=True)

    if count % 5 == 0:
        pred.forecast(n_samples=10, final_time=20_000, verbose=False)

        if 2*pred.forecasted_states_std[-1][0] <= 4700:
            print('Two standard deviations of forecasted crash state below 4.7km; terminating predictor.')
            print(f'Predictor terminated after time {pred.t} seconds.')
            print(f'Final mean forecasted state:\nx = {pred.forecasted_states_mean[-1][0]/1000} km\n')
            print(f'Final forecasted state standard deviation:\nx_std = {pred.forecasted_states_std[-1][0]/1000} km\n')
            print(f'Time difference between confident forecast and real data crash: {times[-1] - pred.t} seconds.')
            break

    else:
        continue
    

import matplotlib.pyplot as plt

plt.figure()
plt.plot(np.asarray(pred.posterior_traj_states)[:,0]/1000, np.asarray(pred.posterior_traj_states)[:,1]/1000, marker='s', label='EKF state')
plt.plot(real_data[:,0]/1000, real_data[:,1]/1000, marker='x', label='Real measurement data')
plt.plot(noisy_data[:,0]/1000, noisy_data[:,1]/1000, marker='.', label='Noisy measurement data ')
plt.xlabel('Distance travelled (km)')
plt.ylabel('Altitude (km)')
plt.legend(loc='best')
plt.show()


# fig, ax = plt.subplots(1, 2)
# ax[0].hist(pred.forecasted_states[0][:,0]/1000, label='First forecast')
# ax[0].set_xlabel('Distance travelled (km)')
# ax[0].legend()

# ax[1].hist(pred.forecasted_states[-1][:,0]/1000, label='Final forecast')
# ax[1].set_xlabel('Distance travelled (km)')
# ax[1].legend()

# plt.show()

