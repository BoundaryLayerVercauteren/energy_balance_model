"""
Script to run randomized version of model by van de Wiel et al.

Van de Wiel, B. J. H., and Coauthors, 2017: Regime transitions in near-surface temperature inversions:
A conceptual model. Journal of the Atmospheric Sciences, 74, 1057–1073, https://doi.org/10.1175/JAS-D-16-0180.1.
"""
import os
import time
import json
import ast
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from DomeC import process_dome_c_data
from one_D_model.utils import plot_output as plot, parse_command_line_input
from one_D_model.model import run_SDE_model as run_1D_SDE_model
from one_D_model.model import parameters, solve_ODE, make_bifurcation_analysis, solve_SDEs, solve_SDE_stoch_stab_function, compare_stability_functions
from one_D_model.model import solve_ODE_with_observation_input


def save_parameters_in_file(params):
    # Define name of parameter file
    file_name = str(params.sol_directory_path) + 'parameters.json'
    # Transform parameter class to json
    params_json = params.to_json()
    # Save json to file + remove unnecessary characters
    with open(file_name, 'w') as file:
        json.dump(ast.literal_eval(params_json), file)


# Read command line input
function, stab_function, Qi, Lambda, z0, u, make_plot, obs_u = parse_command_line_input.read_command_line_input()
# -----------------------------------------------------------------------------------------
# Load Parameters
param = parameters.Parameters()
# -----------------------------------------------------------------------------------------
# Make directory for output
param.sol_directory_path = 'output' + '/' + time.strftime("%Y%m%d_%H%M_%S") + '/'
if not os.path.exists(param.sol_directory_path):
    os.makedirs(param.sol_directory_path)
    os.makedirs(param.sol_directory_path + 'temporary/')
# -----------------------------------------------------------------------------------------
if make_plot:
    # Process Dome C data and plot it
    data_domec = process_dome_c_data.main(param)
    # # -------------------------------------------------------------------------------------
    # # Solve deterministic ODE
    # ODE_sol = solve_ODE.solve_deterministic_ODE(param)
    # # Plot solution of deterministic model
    # plot.make_2D_plot(param, ODE_sol.t.flatten(), ODE_sol.y.flatten(), 'ODE_sol.png')
    # Plot stability functions
    #compare_stability_functions.make_comparison(param)
    # -------------------------------------------------------------------------------------
    # Make bifurcation plots
    # copy dataclass to prevent overwriting original
    param_copy = dataclasses.replace(param)
    param_copy.sol_directory_path = param.sol_directory_path
    param_copy.stab_func_type = 'short_tail'
    make_bifurcation_analysis.make_bifurcation_analysis(param_copy, data_domec)
    param_copy.stab_func_type = 'long_tail'
    make_bifurcation_analysis.make_bifurcation_analysis(param_copy, data_domec)
    # Plot potential
    #plot.plot_potentials(param)
# -----------------------------------------------------------------------------------------
# Run 1D model (with randomizations)
if function:
    run_1D_SDE_model.solve_randomized_model(param)

if Qi:
    function_name = solve_SDEs.solve_SDE_with_stoch_Qi
    sol_file_name = 'SDE_Qi_sol'
    run_1D_SDE_model.solve_model_with_randomized_parameter(param, function_name, sol_file_name)

if Lambda:
    function_name = solve_SDEs.solve_SDE_with_stoch_lambda
    sol_file_name = 'SDE_lambda_sol'
    run_1D_SDE_model.solve_model_with_randomized_parameter(param, function_name, sol_file_name)

if u:
    function_name = solve_SDEs.solve_SDE_with_stoch_u
    sol_file_name = 'SDE_u_sol'
    run_1D_SDE_model.solve_model_with_randomized_parameter(param, function_name, sol_file_name)

if stab_function:
    function_name = solve_SDE_stoch_stab_function.solve_SDE_with_stoch_stab_function
    sol_file_name = 'SDE_stab_func_sol'
    run_1D_SDE_model.solve_model_with_randomized_parameter(param, function_name, sol_file_name)

if z0:
    sol_file_name = 'SDE_z0_sol'
    run_1D_SDE_model.solve_model_with_randomized_parameter_z0(param, sol_file_name)

if obs_u:
    # Load data (10 min averaged)
    observed_u = np.loadtxt('u_values.txt')
    observed_delta_theta = np.loadtxt('delta_theta_values.txt')
    # Set parameters such that they fit with input data
    param.delta_T_0 = observed_delta_theta[0]
    param.t_end_h = int(len(observed_u) / 6)
    param.t_end = param.t_end_h * 3600
    param.dt = 10 * 60
    param.num_steps = int(param.t_end / param.dt)
    param.t_span = np.linspace(param.t_start, param.t_end, param.num_steps)
    param.t_span_h = np.linspace(param.t_start, param.t_end_h, param.num_steps)
    # Solve ODE with u as given by input file
    solution = solve_ODE_with_observation_input.solve_deterministic_ODE(param, observed_u)
    # Save solution
    np.save(param.sol_directory_path + 'solution_ODE_with_observed_u.npy', solution)
    # Plot solution
    color = matplotlib.cm.get_cmap('cmc.batlow', 4).colors
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(param.t_span_h, solution[1], label='model', color=color[0])
    ax1.plot(param.t_span_h, observed_delta_theta[:param.num_steps], label=r'observed $\Delta T$', color=color[1])
    ax1.plot(param.t_span_h, observed_u[:param.num_steps], label='observed u', color=color[2])
    ax1.set_xlabel('time [h]')
    ax1.set_ylabel(r'$\Delta T$ [K]')
    plt.legend()
    plt.savefig(param.sol_directory_path + 'model_vs_observation_over_time.png', bbox_inches='tight', dpi=300)
    # To clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close('all')  # Closes all the figure windows.
# -----------------------------------------------------------------------------------------
# Save parameters
save_parameters_in_file(param)