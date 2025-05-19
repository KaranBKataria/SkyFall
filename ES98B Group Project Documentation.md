# ES98B Group Project Documentation

The GitHub link for the project: https://github.com/KaranBKataria/SkyFall

## Table of Contents

### Introduction

- Primary objectives
- Key package features
- Limitations & current known issues

### Environment & Dependencies

- Environment setup
- Module-wise dependencies
- Package-wise dependencies

### Comprehensive Functionality Guide

- [1. System Architecture](about:blank#1-system-architecture)
- [2. Simulator Module](about:blank#2-simulator-module)
- [3. Predictor Module](about:blank#3-predictor-module)
- [4. Visualiser Module](about:blank#4-visualiser-module)
- [5. Utility Module](about:blank#5-utility-module)

### Other

- Extending functionality
- [Understanding Output Files](about:blank#3-understanding-output-files)

## Introduction

This project provides a **satellite orbital decay simulation and a prediction system**, designed to model and predict the trajectory of a satellite (ISEE-3) entering Earth’s atmosphere along an equatorial orbit.

### Primary objectives:

- Assist **users** in setting up the environment, indicate reasonable use cases and how to launch and vary the simulation to produce visualisations.
- Guide **developers** to understand how to extend the functionality for advanced use cases.

### Key package features:

- Configurable orbital decay simulation based on popular physical models and assumptions:
- Spherical Earth, uniform density, constant rotation.
- Time-invariant satellite surface area, non-rotating around any of its axes.
- U.S. Standard Atmosphere 1976 model.
- Isolated system (Earth and satellite).
- Radar-based state estimation using a configurable **Extended Kalman Filter (EKF)**.

### Limitations & current known issues

- A spherical Earth with uniform density is assumed.
- Assumes constant surface area and a non-tumbling satellite.
- Assumes the drag equation is valid from start to end - does not consider non-continuum effects above 86km.
- Constant radar noise model (no occlusion or range bias).
- EKF-based filtering (prone to numerical instability).
- Uncertainty quantification based on Monte Carlo sampling; computationally expensive for large sample sizes.

## Environment & Dependencies

### Environment setup

To ensure smooth execution and modular usage of the project, please follow the environment setup guidelines below. The system should be compatible across platforms, but this has not been rigorously tested.

---

**Recommended Global Environment**

| Item | Recommended Configuration |
| --- | --- |
| Python Version | Python ≥ 3.10 |
| Operating System | Windows 10/11, MacOS, or Linux |
| Package Manager | `pip` or `conda` |

### Module-wise dependencies

The project is designed with a modular, class-based architecture, allowing individual components (e.g., the simulator, predictor, or visualiser) to be executed or extended independently.

We explicitly state the dependency versions to ensure long-term maintainability, prevent future compatibility issues, and support modular reuse or integration in other systems.

| Module | Path | Python Version | Notes |
| --- | --- | --- | --- |
| **Simulator** | `SkyFall/simulator/` | **≥ 3.10** | Uses `NumPy`, `SciPy`; any version ≥3.10 is sufficient for numerical solvers |
| **Predictor** | `SkyFall/predictor/` | **3.x** | EKF, compatible with any modern Python 3 version |
| **Visualiser** | `SkyFall/visualisation/` | **≥ 3.10** | Uses `matplotlib`, `cartopy` (include link); tested with 3.10+ |
| **Utils** | `SkyFall/utils/` | – | Shared functions, no specific requirements |

### Package-wise dependencies

This project relies on several third-party Python packages for numerical integration, state estimation, visualisation and much more. Below is a complete list of dependencies used across all modules, along with module-specific usage notes and special considerations.

**Required Packages**

To use SkyFall, the following third-party packages are required. For the full list of required dependencies and their versions, please see `requirements.txt`. 

| Package | Purpose | Used In Modules |
| --- | --- | --- |
| `numpy` | Core numerical computation, array operations | Multiple modules |
| `scipy` | ODE solver (`solve_ivp`) | `simulator`, `utils`, `visualiser` |
| `matplotlib` | Static plotting | `visualiser` |
| `tqdm` | Progress bar for loops | `visualiser`, `simulator` |
| `sympy` | Symbolic matrix construction (Jacobians) | `utils/analytical_F.py`, `utils/analytical_H.py` |

Note: built-in standard libraries such as `os`, `sys`, `datetime`, `time`, `math`, and `json` are also used across multiple modules, but do not require separate installation.

# Comprehensive Functionality Guide

This section is intended for understanding how to modify or extend the internal structure of the system.

It describes:

- The package architecture and the data exchanges between modules.
- Key functions within each module and their customisation.

### 1. [System Architecture](about:blank#1-system-architecture)

```
SkyFall
│
├── main.py                     # Example script: run simulation and prediction
├── requirements.txt            # Python dependency list
├── README.md                   # Project overview and instructions
│
├── SkyFall/                    # Main source package
│
│   ├── simulator/
│       └── simulator.py        # Orbital dynamics + radar measurement simulation
│
│   ├── predictor/
│       └── predictor.py        # EKF state estimation and impact prediction
│       └── __init__.py         # Python package initializer
|
│   ├── visualiser/
│       └── visualiser.py       # Trajectory plotting
│
│   ├── utils/
│       ├── global_variables.py    # Global constants (G, Re, ω, etc.)
│       ├── analytical_F.py        # ∂f/∂x Jacobian for EKF state transition
│       ├── analytical_H.py        # ∂h/∂x Jacobian for EKF measurement model
│       ├── predictor_utilities.py # State propagation + RK45 for EKF
│       └── __init__.py            # Python package initializer
│
│   └── __init__.py             # Python package initializer

```

| Module | Responsibility |
| --- | --- |
| `simulator/` | Simulates satellite motion and radar observations |
| `predictor/` | Applies EKF to estimate trajectory and impact site via Monte Carlo sampling |
| `visualisation/` | Generates plots and terminal outputs from prediction |
| `utils/` | Stores shared constants, symbolic Jacobians, etc. |

The main script `main.py` calls these modules sequentially, handling data flow and integration logic. See tutorial for a worked example.

1. Using Individual Modules

---

### 2. `Simulator.py`:

**Simulator (Class)**

- inputs:
    - `initial_state`: np.array,
        - This specifies the starting point of the satellite from the moment it begins experiencing orbital decay.
    - `measurement_covariance`: np.array,
        - This specifies a constant matrix R. This is the noise corrupting the truth of measured attributes of the satellite.
    - `timestep`: float,
        - - Timestep for the numerical method RK45.
    - `t0`: float = 0,
        - Starting point in time.
    - `t_end`: float = 4e9
        - An unreasonably large value (~127 years) on which to end the simulation. This ensures the measurements till the satellite crashes are captured.

**get_measurements (function)**

- inputs:
    - `Verbose`: Bool
        - This signposts the calling, the beginning and end of execution of the function.
- outputs:
    - `times`: np.array,
        - The times for which the ODE was numerically stepped through.
    - `measurement_noiseless`: np.array,
        - The 'real' radar station measurements (i.e. no noise).
    - `measurement_noise`: np.array,
        - The radar station measurements perturbed with additive Gaussian noise.
    - `active_radar_stations_per_time`: np.array,
        - A history of active radar stations per time step by index.
    - `active_radar_station_longitudes_per_time`: np.array,
        - A history of active radar stations' longitudes per time.
    - `crash_site`: np.array,
        - Given the tranquillity base condition on the computed standard deviation of the periodic ODE propagated Monte Carlo trajectories, this would be their mean at r=R_E.
    - `crash_time[0]`: float,
        - This is the time elapsed since the start of the simulation, added by the time at which the tranquillity base condition was satisfied by the ODE propagated MC trajectories.
    - `state_noiseless.T`: np.array,
        - The trajectory of the state vector through the dynamics of the ODE system.

---

### 3. `Predictor.py`:

**Predictor (class)**

**process_model (function):** Obtain the prior state at time $t$, mean by passing the posterior random variable, from time $t-\Delta t$ through the ODE and a sample is taken from a Gaussian random variable, $\sim N(\vec{0},Q)$, and added to it.

- Inputs:
    - `posterior_state`: np.array
        - This is the posterior state, $\vec{x}_{t-\Delta t}$, obtained at the end of the last time’s iteration through the EKF algorithm.
    - `process_covariance`: np.array
        - This is the process covariance matrix $Q$. This is a measure of our ‘belief’ in the physics/ in the validity of the model of the dynamics.
    - `timestep`: float
        - This is the timestep used in the RK45 numerical finite difference method to transition from $\vec{x_{t}}$ to $\vec{x}_{t+\Delta t}$ through the continuous physics ODE dynamics.
    - `t`: float
        - This is the time elapsed until the last time step was taken.
    - `include_noise`: bool
        - If true, sample once from $Q$ and add to $\vec{x}_{t+\Delta t}$ else add $0$.
    - `verbose`: Bool
        - If true, print $\vec{x}_{t+\Delta t}$, else print nothing.
- Outputs:
    - `prior_state`: np.array
        - This is updated from equalling posterior, $\vec{x}*{t-\Delta t}$, to equalling the posterior passed through discretised dynamics, $\bar{x}*{t}$.

**eval_JacobianF() (function):** Leverages analytical_F.py’s access to closed-form matrix of partial derivatives, via SymPy for the continuous ODE system. Used to propagate old posterior state covariance, $P_{t-\Delta t}$, to the prior state covariance, $\bar{P}_{t}$

- Inputs:
    - `G, M_e, Cd, A, m, R_star, g0, M_molar, omega_E`: floats
        - See 5. Utility Module for an explanation of these empirically found constants.
- Outputs:
    - $F_{t} := \left.\frac{\partial f(\vec{x})}{\partial \vec{x}}\right|*{ \vec{x}*{t-\Delta t}}$where f(.) is the dynamics of the ODE system for satellite orbital decay.

**update_prior_belief() (function):** Incorporates uncertainty in equations of motion and the current state’s computed process matrix to find the covariance matrix of the predicted prior state vector

- Inputs:
    - `JacobianF`: np.array
        - Output of evaluation at current iteration of eval_JacobianF().
    - `process_covariance`: np.array
        - This is the process covariance matrix $Q$. This is a measure of our ‘belief’ in the physics/ in the validity of the model of the dynamics.
    - `posterior_state_covariance`: np.array
        - This is the previous EKF iteration’s posterior state’s covariance $P_{t-\Delta t}$, this was computed previously by evaluating $(I-K_{t-\Delta t}H_{t-\Delta t})\bar{P}*{t-\Delta t}$ where $K*{...}$ and $H_{...}$ are defined below at the current iteration, t.
    - `JacobianV` : np.array
        - See the Functionality Extension for significance, but the package as is sets it to the zero matrix.
    - `verbose`: Bool
        - If true, print $\bar{P}*t = F_t \,P*{t-\Delta t}\, F_t^T + Q$ else nothing.
- Outputs:
    - $\bar{P}*t := F_t \,P*{t-\Delta t}\, F_t^T + Q$, we call this the current prior state covariance matrix and it is a result of incorporating the physics model uncertainty in the previous iteration’s posterior state after it has been propagated through the current iterations’ system dynamics.

**residual() (function): (function):**  Using the measurement_model() we transform the current prior state to measurement space.

**eval_JacobianH() (function):** Uses and evaluates at the current prior state, the symbolic Jacobians from `utils/.`

- Inputs:
    - `theta_R`:
        - The longitude of the ‘active’ satellite at the time self.t
    - `R_e, omega_E`: floats
        - See \utils for these empirically found constants
- Outputs:
    - $H_{t} = \left.\frac{\partial h(\vec{x})}{\partial \vec{x}}\right|*{ \vec{x}*{t}}$

**measurement_model() (function):** Converts state into measurement space for residual calculation.

- Inputs:
    - `prior_state`: np.array
        - This is the (with noise/ without noise) output of the process model.
    - `theta_R`: float
        - This is the longitude of the ‘active’ satellite determined by the simulator during the satellite descent.
    - `t`: float
        - The current time in the Earth-satellite system.
- Outputs:
    - `h` :
        - The projection of the state vector into measurement space, $[r,\dot{r}]$.

**kalman_gain() (function):** Kalman update step, which will act to scale the residual based on the strength of belief in the measurement.

**predict_state() (function):** Uses Kalman gain matrix to combine residual with predicted prior estimate to form a mean predicted state vector and corresponding covariance matrix.

**forecast() (function):** Samples are possible from the current posterior of the predicted state vector and propagate them using a numerical method to ‘project’ the current predicted posterior on Earth’s surface.

**run_predictor() (function):** This function serves as a wrapper for running the predictor until termination, obtaining information about its estimated trajectories, impact distribution and statistics, etc. This is a user-friendly alternative to manually creating a loop for the predictor.

- Inputs:
    - `predictor`: class object
        - An instantiated object of the `Predictor` class
    - `radar_measurements`: np.array
        - An array of noisy radar station measurements from the simulator
    - `active_radar_longitudes`: np.array
        - An array of the longitudes of the radar stations which provided the measurements, also given by the simulator
    - `num_samples_MC`: int
        - Number of Monte Carlo samples to draw in the forecast step
    - `forecast_gap`: int
        - The number of measurements between forecast steps
    - `verbose`: bool
        - A boolean value to output print statements from the predictor

---

### 4. `Visualiser.py`

The `Visualiser` class visualises satellite orbit decay using Cartopy maps, height vs time plots, and crash longitude distributions. It supports static plots, animations, and customizable styles.

**Visualiser (Class)**

- **inputs**:
    - `times`: np.array
        - Time points (in seconds) corresponding to the satellite’s trajectory.
    - `trajectory_cartesian`: np.array
        - Satellite trajectory in Cartesian coordinates (x, y, vx, vy) in meters and meters per second.
    - `trajectory_LLA`: np.array
        - Satellite trajectory in Latitude, Longitude, Altitude (LLA) coordinates, with latitude and longitude in degrees and altitude in meters.
    - `crash_lon_list`: np.array
        - List of predicted crash longitudes (in degrees) for each forecast step.

`_adjust_crash_lon_list` (Function)

- **inputs**:
    - `crash_lon_list`: np.array
        - Array of crash longitude predictions with shape [sets, samples, dims].
    - `target_length`: int
        - Desired length of the crash longitude list, equal to the length of times.
- **outputs**:
    - `adjusted_list`: np.array
        - An array of crash longitudes adjusted to match the target length by repeating or extending prediction sets.

`plot_height_vs_time` (Function)

- **inputs**:
    - `figsize`: tuple = (8, 6)
        - Figure size (width, height) in inches.
    - `title`: str = 'Predictor altitude estimate against time'
        - Title of the plot.
    - `title_fontsize`: int = 14
        - Font size for the plot title.
    - `label_fontsize`: int = 12
        - Font size for axis labels.
    - `tick_fontsize`: int = 10
        - Font size for tick labels.
    - `line_color`: str = 'blue'
        - Color of the altitude line.
    - `line_width`: float = 2
        - Width of the altitude line.
    - `show_grid`: bool = True
        - Whether to display a grid on the plot.
    - `show_legend`: bool = True
        - Whether to display a legend.
- **outputs**:
    - None
        - Displays a plot of altitude versus time with a termination line.

`plot_orbit_map` (Function)

- **inputs**:
    - `figsize`: tuple = (8, 6)
        - Figure size (width, height) in inches.
    - `title`: str = 'Satellite Orbital Decay Trajectory'
        - Title of the plot.
    - `title_fontsize`: int = 14
        - Font size for the plot title.
    - `tick_fontsize`: int = 10
        - Font size for tick labels.
    - `path_color`: str = 'red'
        - Color of the orbit path.
    - `start_marker_color`: str = 'green'
        - Color of the start position marker.
    - `end_marker_color`: str = 'red'
        - Color of the end position marker.
    - `marker_size`: int = 10
        - Size of start and end markers.
    - `scatter_size`: int = 50
        - Size of scatter points representing altitude.
    - `cmap`: str = 'viridis'
        - Colormap for altitude scatter points.
    - `show_legend`: bool = True
        - Whether to display a legend.
- **outputs**:
    - None
        - Displays a Cartopy map showing the satellite’s orbit path with altitude-colored scatter points.

`plot_crash_distribution` (Function)

- **inputs**:
    - `figsize`: tuple = (8, 6)
        - Figure size (width, height) in inches.
    - `title`: str = 'Predicted crash site distribution'
        - Title of the plot.
    - `title_fontsize`: int = 14
        - Font size for the plot title.
    - `label_fontsize`: int = 12
        - Font size for axis labels.
    - `tick_fontsize`: int = 10
        - Font size for tick labels.
    - `box_color`: str = 'blue'
        - Color of the boxplot.
    - `show_grid`: bool = True
        - Whether to display a grid on the plot.
- **outputs**:
    - None
        - Displays a boxplot of predicted crash longitudes versus forecast index.

`plot_orbit` (Function)

- **inputs**:
    - `figsize`: tuple = (12, 10)
        - Figure size (width, height) in inches.
    - `title_fontsize`: int = 14
        - Font size for subplot titles.
    - `label_fontsize`: int = 12
        - Font size for axis labels.
    - `tick_fontsize`: int = 10
        - Font size for tick labels.
    - `map_title`: str = 'Satellite orbital decay trajectory'
        - Title for the map subplot.
    - `height_title`: str = 'Altitude against time'
        - Title for the height versus time subplot.
    - `crash_title`: str = 'Predicted crash site distribution'
        - Title for the crash distribution subplot.
    - `path_color`: str = 'red'
        - Color of the orbit path.
    - `height_line_color`: str = 'blue'
        - Color of the altitude line.
    - `box_color`: str = 'blue'
        - Color of the crash distribution boxplot.
    - `show_grid`: bool = True
        - Whether to display grids on the subplots.
    - `show_legend`: bool = True
        - Whether to display legends on the subplots.
- **outputs**:
    - None
        - Displays a 2x2 grid with a Cartopy map, altitude versus time plot, and crash longitude distribution plot.

`save_plot` (Function)

- **inputs**:
    - `filename`: str = 'orbit_decay.png'
        - Output file name for the saved plot.
    - `figsize`: tuple = (12, 10)
        - Figure size (width, height) in inches.
    - `title_fontsize`: int = 14
        - Font size for subplot titles.
    - `label_fontsize`: int = 12
        - Font size for axis labels.
    - `tick_fontsize`: int = 10
        - Font size for tick labels.
    - `map_title`: str = 'Satellite Orbit Decay Path'
        - Title for the map subplot.
    - `height_title`: str = 'Height vs Time'
        - Title for the height versus time subplot.
    - `crash_title`: str = 'Predicted Crash Longitude Distribution'
        - Title for the crash distribution subplot.
    - `path_color`: str = 'red'
        - Color of the orbit path.
    - `height_line_color`: str = 'blue'
        - Color of the altitude line.
    - `box_color`: str = 'blue'
        - Color of the crash distribution boxplot.
    - `show_grid`: bool = True
        - Whether to display grids on the subplots.
    - `show_legend`: bool = True
        - Whether to display legends on the subplots.
    - `dpi`: int = 300
        - Resolution for the saved image.
    - `bbox_inches`: str = 'tight'
        - Bounding box setting for saving the plot.
- **outputs**:
    - None
        - Saves a 2x2 grid plot (map, height, crash distribution) to the specified file.

`animate_orbit` (Function)

- **inputs**:
    - `interval`: float = 50
        - Time between animation frames in milliseconds.
    - `figsize`: tuple = (12, 12)
        - Figure size (width, height) in inches.
    - `title_fontsize`: int = 14
        - Font size for subplot titles.
    - `label_fontsize`: int = 12
        - Font size for axis labels.
    - `tick_fontsize`: int = 10
        - Font size for tick labels.
    - `map_title`: str = 'Satellite Orbital Decay Animation'
        - Title for the map subplot.
    - `height_title`: str = 'Altitude against time'
        - Title for the height versus time subplot.
    - `crash_title`: str = 'Predicted crash site distribution'
        - Title for the crash distribution subplot.
    - `path_color`: str = 'red'
        - Color of the orbit path.
    - `current_point_color`: str = 'red'
        - Color of the current position marker.
    - `height_line_color`: str = 'blue'
        - Color of the altitude line.
    - `crash_point_color`: str = 'black'
        - Color of crash site points on the map.
    - `crash_box_color`: str = 'blue'
        - Color of the crash distribution boxplot.
    - `marker_size`: int = 8
        - Size of markers for the current position and start point.
    - `scatter_size`: int = 50
        - Size of scatter points for altitude.
    - `cmap`: str = 'viridis'
        - Colormap for altitude scatter points.
    - `show_grid`: bool = True
        - Whether to display grids on the subplots.
    - `show_legend`: bool = True
        - Whether to display legends on the subplots.
    - `button_pos_replay`: list = [0.45, 0.05, 0.1, 0.05]
        - Position and size of the replay button [x, y, width, height].
- **outputs**:
    - None
        - Displays an animated 2x2 grid with a Cartopy map, altitude versus time plot, and crash longitude distribution, including a replay button.

---

### 5. Utility Module

**Purpose**: Provides reusable constants, symbolic Jacobians, and helpers.

**Key Files**

- `global_variables.py`:
    - `predictor_termination`: float = $0.0007377 \mathrm{rad}$
        - Termination criterion for the predictor based on the Tranquillity base
    - `G`: float = $6.6743×10^{-11}$ $\mathrm{m^3 kg^{-1}s^{-2}}$
        - Gravitational constant
    - `M_e:` float = $5.972 × 10² \mathrm{kg}$
        - Mass of the Earth
    - `C_d`: float = $2.2$
        - Drag Coefficient
    - `A`: float = $\pi (1.77/2)^2 \mathrm{m}^2$
        - Area of the satellite
    - `m_s`: float = $479 \mathrm{kg}$
        - Mass of the satellite
    - `R_star`: float = $8.3144598 \mathrm{J mol^{-1}K^{-1}}$
        - Universal ideal gas constant
    - `g_0:`  float = $9.80665 \mathrm{m s^{-2}}$
        - Gravitational acceleration at Earth’s surface
    - `M_molar`: float = $0.0289644 \mathrm{kg mol^{-1}}$
        - Molar mass of Earth’s air
    - `omega_E`: float = $7.2921150 × 10⁻⁵ \mathrm{rad s^{-1}}$
        - Angular rotation rate of the Earth
- `predictor_utilities.py`:
    - covariance_matrix_initialiser(): np.ndarray
        - Initialises a valid covariance matrix from diagonal variances (and optional off-diagonal covariances).
            - **Inputs**
                - `variances` (`np.ndarray` of shape `(n,)`): variances along the diagonal
                - `covariances` (`None` or `np.ndarray` of length `n(n−1)/2`): upper-triangle values
            - **Output**
                - `cov_mat` (`(n,n) np.ndarray`): symmetric covariance matrix
    - USA76_air_density(): float
        - Returns the ambient air density at altitude `y` (m) using the COESA-76 model.
            - **Inputs**
                - `y` (`float`): altitude above mean Earth radius (m)
            - **Output**
                - `rho` (`float`): density (kg m⁻³)
            - **Details**
                - Internally calls `coesa76(y_km)` from pyatmos for all altitudes
                - Falls back to barometric layers below 86 km if needed
    - equations_of_motion(): List[float]
        - Evaluates the right–hand side of the polar‐ECI equations of motion at one time step.
            - **Inputs**
                - `time` (`float`): current simulation time (s)
                - `state` (`np.ndarray` of shape `(4,)`): `[r, θ, ṙ, θ̇]`
            - **Output**
                - `f` (`List[float]`): derivatives `[ṙ, θ̇, r̈, θ̈]`
    - hit_ground(): float
        - Event function for ODE solver: triggers when the satellite radius `r` equals Earth radius.
            - **Inputs**
                - `time` (`float`): current time (unused)
                - `state` (`np.ndarray`):
            - **Output**
                - `value` (`float`): `r − R_e`; zero indicates ground impact
            - **Attributes**
                - `terminal = True` (stop integration)
                - `direction = -1` (only when `r` decreasing)
    - longitude_cal(): float
        - Converts an equatorial arc‐length (m) into longitude (°E).
            - **Inputs**
                - `distance` (`float`): eastward distance along equator (m)
            - **Output**
                - `longitude` (`float`): corresponding longitude in degrees (–180, +180)
    - polar_to_cartesian_state(): np.ndarray
        - Transforms a state vector in polar `[r, θ, ṙ, θ̇]` into Cartesian `[x, y, ẋ, ẏ]`.
            - **Inputs**
                - `state` (`np.ndarray` of length 4)
            - **Output**
                - `cartesian_state` (`np.ndarray` of length 4)
    - measurement_model_h(): np.ndarray
        - Maps a polar ECI state to the two radar measurements: range & range‐rate.
            - **Inputs**
                - `state` (`np.ndarray` of length 4): `[r, θ, ṙ, θ̇]`
                - `radar_longitude` (`float`): station longitude in ECI (rad)
            - **Output**
                - `h` (`np.ndarray` of length 2): `[ρ, ṙₑ]`
    - physical_quantities(): np.ndarray
        - Computes derived “physical” outputs for visualisation.
            - **Inputs**
                - `state` (`np.ndarray`): current `[r, θ, ṙ, θ̇]`
                - `initial_state` (`np.ndarray`)
            - **Output**
                - `physical_state` (`np.ndarray` of length 4):`[distance_along_equator, altitude, v_x, v_y]`
    - ECI_to_ECEF(): Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - Converts a polar ECI state to ECEF polar & Cartesian plus LLA.
            - **Inputs**
                - `time` (`float`): current simulation time (s)
                - `state` (`np.ndarray`): polar ECI `[r, θ, ṙ, θ̇]`
            - **Outputs**
                - `state_ECEF` (`np.ndarray`): polar ECEF `[r, θ_e, ṙ, θ̇_e]`
                - `state_ECEF_cartesian` (`np.ndarray`): Cartesian ECEF `[x, y, ẋ, ẏ]`
                - `LLA_radians` (`np.ndarray`): `[lat, lon, alt]` in radians & m
                - `LLA_deg` (`np.ndarray`): `[lat, lon, alt]` in degrees & m
    
- `analytical_F.py`, `analytical_H.py`calculate analytical Jacobians of the process and measurement models using `sympy` for symbolic differentiation. These output lambda functions, using sympy, of the analytically derived Jacobians.

**Customisation Tips**

- Change Earth or model constants in `global_variables.py`.
- Edit symbolic expressions if the model dynamics are altered.

## Extending Functionality

### Custom Radar-Station Placement and Visibility

This section describes in-depth how to extend the simulator and predictor to work with user-defined, geographically specific radar stations. 

- Defining Coverage Arcs:
    - `land_arcs_deg`: List[Tuple[float,float]]
        - Each tuple `(lon_start, lon_end)` defines an interval in degrees East of the equator where stations may be placed.
        - Wrap‑around across the equator: if `lon_end < lon_start`, the interval wraps via ±180°.
        - Allocating the Radar Station Counts Proportionally:
            - Compute each arc’s angular length in degrees.
            - For each span `L`, allocate the radar stations proportionally by setting `N_i = round(N_total * L/total_length)`
            - Adjusting the rounding: by letting `delta = N_total - sum(N_i)` and by adding `delta` to the arc with the largest span.
            - Uniform Station Spacing within Arcs:
                - For each arc and its allocated count `n_i`
- Uniform Radar Station Spacing within Arcs
    - This step takes each land-arc interval and its allocated station count `n_i` and places stations at equal angular intervals along the arc. For an arc from `lo` to `hi`, compute:
    `lons = np.linspace(lo, lo + ((hi - lo) % 360), n_i)`
    - The longitudes are then normalised into the $[-180 \degree, +180 \degree)$  and converted to radians for ECI calculations.
    - For further customisability, replace the linear spacing with weighted sampling (concentrating more stations in high-risk regions).
- Visibility Half-Angle $\beta$ Calculation:
    - Assuming a maximum altitude `h_star` from which a station can still see the satellite.
    - The geometry produces: $\beta = \arccos \frac{R_E}{R_E + h_{star}}$
- Active Station Selection: `active_station`
    - This function first rotates each station’s fixed longitude into the ECI frame by accounting for Earth’s rotation, then computes the smallest signed angular separation between the satellite and each station.
    - Inputs:
        - `lon_sat`: satellite longitude (rad) in ECI frame
        - `t`: simulation time (s)
    - Outputs:
        - `idx`: index of nearest station within coverage
        - `diff`: angular offset
        

### Provision of a control input (thrust) into the predictor

This section outlines how the functionality of the predictor can be enhanced by including a control input, i.e. thrust, into the motion of the satellite. The provision of a control input enables the satellite to activate its thrust to prevent a crash landing in a populated area; such a feature has obvious merits regarding safety.

The `Predictor` class currently has existing functionality, particularly in the `update_prior_belief` function, to support the inclusion of a control input by allowing the user to specify a control covariance matrix $\mathbf{V}$. This is currently set to a default value of `None`, but can be created via the `covariance_matrix_initialiser` function found in `predictor_utils.py` and passed in.