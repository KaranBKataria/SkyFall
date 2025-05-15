
## Getting started

To use the package, clone this repository onto your local environment by typing the following command into your shell:

```bash
# Clone the repository
git clone https://github.com/KaranBKataria/ES988-ISEE.git
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

## Example script

To see an example script of how to use SkyFall and the output, inspect `main.py`. To see the output, run the script via the following command in the shell

```bash
python3 main.py

# Alternatively
python main.py
```
