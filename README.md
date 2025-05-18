
# SkyFall

SkyFall is a Python-based library for simulating and predicting the crash site of a de-orbiting satellite along the equator of a spherical, rotating Earth. The package is based on the [ISEE-3](https://science.nasa.gov/mission/isee-3-ice/) satellite by default.

Please see the documentation markdown file for more details about the underlying functionality, arguments, assumptions, and more. 

## Getting started

To use the package, clone this repository onto your local environment by typing the following command into your shell:

```bash
# Clone the repository
git clone https://github.com/KaranBKataria/SkyFall.git
```

Once cloned, change directory into the `SkyFall` folder and create a Python virtual environment (via venv) to prevent crashing dependencies:

```bash
# Navigate into the directory
cd SkyFall

# Create a virtual environment
python3 -m venv venv

# Alternatively
python -m venv venv

# Activate Python virtual environment
source venv/bin/activate (MacOS/Linux)
venv\Source\activate.bat (Windows)
```

> [!IMPORTANT]  
Running the commands `python3 -m` or `python -m` assumes that Python is located in the `PATH` environment variable. Please ensure this is the case before running the commands above. To check whether Python is within `PATH`, run the following command `echo $PATH` (MacOS/Linux) or `echo %PATH%` (Windows).

Once set up and activated, install all required dependencies via the `requirements.txt` file by running the following command:

```bash
pip3 install -r requirements.txt

# Alternatively,
pip install -r requirements.txt
```

This will install all required dependencies and versions without causing clashes across other projects.

Begin writing a driver script, importing the package by creating a `.py` file in the top-level `SkyFall` directory, for example:

```bash
touch user_scrip.py
```

Once created, the user can import the functionalities of the package using the following:

```python
# Import the utility functions required for both the simulator and the predictor
from SkyFall.utils import predictor_utilities

# Import all global constant variables (e.g. Mass of the Earth)
from SkyFall.utils.global_variables import *

# Import in the simulator
from SkyFall.simulator.simulator import Simulator

# Import in the predictor
from SkyFall.predictor import Predictor
```

>[!NOTE]
Although not common practice, all global variables have been imported using the snowflake command `*`—this is intentional, as it floods the namespace with essential global variables used throughout the package without calling, for example, `global_variables.M_e` or other aliases alike.

## Usage

SkyFall takes a modular, class-based approach to simulating the de-orbit dynamics and predicting the crash site of the satellite. The package is broken up into the following modules: `simulator`, `predictor` and `visualiser`, with utility functions and global variables housed in the `utils` module.

Below is a step-by-step guide on how to initialise, call and use the modules once imported as shown in the **Getting started** section.

### Preliminaries: user-specified arguments

The three modules share many common user-specified inputs. Therefore, it is required for the user to define such shared parameters before instantiating objects of the modules. Below are the common variables required, as well as functionality to create them (please see the documentation markdown file for the mathematical insight and more details on functions and their associated arguments):

```python
# Number of forecast Monte Carlo samples 
n_samples: int = ...

# Number of measurements between forecasts made
nth_measurement: int = ...

# Define the covariance matrices

# State covariance matrix - shape (4,4)
P: np.array = predictor_utilities.covariance_matrix_initialiser(variances=[...], covariances=[...])

# Measurement covariance matrix - shape (2,2)
R: np.array = predictor_utilities.covariance_matrix_initialiser(variances=[...], covariances=[...])

# Process covariance matrix - shape (4,4)
Q: np.array = predictor_utilities.covariance_matrix_initialiser(variances=[...], covariances=[...])

# Initial state must be an array of shape (4,)
x0: np.array = np.array(...)

# Time step between radar measurements and EKF predictions (seconds)
del_t: float = ...

# Initial time (seconds)
t0: float = ...
```

>[!NOTE]
> The initial state must be defined in the following order: radial distance from the centre of the Earth to the satellite, initial longitude of the satellite, radial velocity and angular velocity. This ordering extends to defining the variances and covariances of the process and state covariance matrices. All units are in metres, radians and seconds.
>
### Simulator

To obtain radar measurements, the user must instantiate the `Simulator` class to generate radar station measurements. This can be achieved in the following way.

```python
simulator = Simulator(initial_state=x0, measurement_covariance=R, timestep=del_t, t0=t0)

times, real_measurements, noisy_measurements, active_radar_indices, active_radar_longitudes, crash_site, crash_time, full_trajectory = simulator.get_measurements()
```

The primary outputs of interest to pass into the predictor include `noisy_measurements` and `active_radar_longitudes`, which are the noisy radar station measurements (radial distance and velocity of the satellite from the radar station) as well as the longitude of the radar station which captured the measurement.

To give the user the flexibility to conduct further analysis, outputs such as the impact site and times, full trajectory, as well as the times are made available. It is key to recall that the simulator emulates radar measurements; having access to the dynamics solved by the simulator enables the user to gain insight into the satellite's true motion. 

### Predictor

Having obtained the radar measurements, the user can instantiate an object of the `Predictor` class to estimate the position of the satellite at each specified time step and obtain a distribution of impact site and time forecasts via Monte Carlo sampling. The predictor is based on the Extended Kalman Filter (EKF) algorithm; please see the documentation for further mathematical insight.

```python
# Create an instance of the predictor
predictor = Predictor(process_covariance=Q, measurement_covariance=R, state_covariance=P, initial_state=x0, timestep=del_t, t0=t0)
```

<img src="https://postimg.cc/G45YQHH8" alt="Alt text" width="400"/>

## Example script

To see an example script of how to use SkyFall and the output, inspect `main.py`. To see the output, run the script via the following command in the shell

```bash
python3 main.py

# Alternatively
python main.py
```
