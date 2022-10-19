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
from one_D_model.utils import plot_output as plot
from one_D_model.model import run_SDE_model as run_1D_SDE_model
from one_D_model.model import parameters, solve_ODE, make_bifurcation_analysis


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
# -----------------------------------------------------------------------------------------
# Make directory for output
param.sol_directory_path = 'output' + '/' + time.strftime("%Y%m%d_%H%M_%S") + '/'
if not os.path.exists(param.sol_directory_path):
    os.makedirs(param.sol_directory_path)
# -----------------------------------------------------------------------------------------
# Save parameters
save_parameters_in_file(param)
# -----------------------------------------------------------------------------------------
# Process Dome C data and plot it
#process_dome_c_data.main(param)
# -----------------------------------------------------------------------------------------
# # Solve deterministic ODE
# ODE_sol = solve_ODE.solve_deterministic_ODE(param)
# # Plot solution of deterministic model
# plot.make_2D_plot(param, ODE_sol.t.flatten(), ODE_sol.y.flatten(), 'ODE_sol.png')
# # Plot potential
# plot.plot_potentials(param)
# # -----------------------------------------------------------------------------------------
# # Make bifurcation plots
# make_bifurcation_analysis.make_bifurcation_analysis(param)
# -----------------------------------------------------------------------------------------
# Run 1D model (with randomizations)
run_1D_SDE_model.main(param)
