"""
Script to run randomized version of model by van de Wiel et al.

Van de Wiel, B. J. H., and Coauthors, 2017: Regime transitions in near-surface temperature inversions:
A conceptual model. Journal of the Atmospheric Sciences, 74, 1057–1073, https://doi.org/10.1175/JAS-D-16-0180.1.
"""
import os
import time
import json
import ast

from DomeC import process_dome_c_data
from one_D_model.model import run_model as run_1D_model
from one_D_model.model import parameters


def save_parameters_in_file(params):
    # Define name of parameter file
    file_name = str(params.sol_directory_path) + 'parameters.json'
    # Transform parameter class to json
    params_json = params.to_json()
    # Save json to file + remove unnecessary characters
    with open(file_name, 'w') as file:
        json.dump(ast.literal_eval(params_json), file)


# Load Parameters
param = parameters.Parameters()

# Make directory for output
param.sol_directory_path = 'output' + '/' + time.strftime("%Y%m%d_%H%M_%S") + '/'
if not os.path.exists(param.sol_directory_path):
    os.makedirs(param.sol_directory_path)

# Save parameters
save_parameters_in_file(param)

# Process Dome C data and plot it
#process_dome_c_data.main(param)

# Make bifurcation analysis and run 1D model (with and without randomizations)
run_1D_model.main(param)
