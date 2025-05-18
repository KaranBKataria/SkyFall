
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

### Preliminaries: user-specified arguements

The three modules share many common user-specified inputs. Therefore, it is required for the user to define such shared parameters before instantiating objects of the modules. Below are the common variables required, as well as functionality to create them (please see the documentation markdown file for the mathematical insight and more details on functions and their associated arguments):

```python
# Number of forecast Monte Carlo samples 
n_samples: int = ...

# Number of measurements between forecasts made
nth_measurement: int = ...

# Define the covariance matrices

# State covariance matrix (shape 4x4)
P: np.array = predictor_utilities.covariance_matrix_initialiser(variances=[...], covariances=[...])

# Measurement covariance matrix (shape 2x2)
R: np.array = predictor_utilities.covariance_matrix_initialiser(variances=[...], covariances=[...])

# Process covariance matrix (shape 4x4)
Q: np.array = predictor_utilities.covariance_matrix_initialiser(variances=[...], covariances=[...])

# Initial state must be an array of shape 4x1
x0: np.array = np.array(...)

# Time step between radar measurements and EKF predictions
del_t: float = ...

# Initial time
t0: float = ...

```

### Simulator

To use the simulator, the user must instantiate the `Simulator` class to generate radar station measurements. This can be achieved in the following way.

```python
simulator = Simulator(initial_state=x0, measurement_covariance=R, timestep=del_t, t0=t0)
```

## Example script

To see an example script of how to use SkyFall and the output, inspect `main.py`. To see the output, run the script via the following command in the shell

```bash
python3 main.py

# Alternatively
python main.py
```
