#!/usr/bin/env python
"""
Main script to run energy balance model (with perturbations).
"""
import os
import sys
import time
import json
import ast
import itertools
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# To be able to run this script on fox
sys.path.append(os.getcwd())

from DomeC import process_dome_c_data
from one_D_model.utils import plot_output as plot, parse_command_line_input
from one_D_model.model import run_SDE_model as run_1D_SDE_model
from one_D_model.model import parameters, solve_ODE, make_bifurcation_analysis, solve_SDEs, \
    solve_SDE_stoch_stab_function, compare_stability_functions


def save_parameters_in_file(params):
    # Define name of parameter file
    file_name = str(params.sol_directory_path) + 'parameters.json'
    # Transform parameter class to json
    params_json = params.to_json()
    # Save json to file + remove unnecessary characters
    with open(file_name, 'w') as file:
        json.dump(ast.literal_eval(params_json), file)


def run_model(param, function=False, stab_function=False, Qi=False, Lambda=False, u=False, make_plot=False,
              stab_function_multi_noise=False, u_and_function=False, stab_function_multi_noise_u_td=False):
    # If command line flag is given make plots
    if make_plot:
        # # Process Dome C data and plot it
        # data_domec = process_dome_c_data.main()
        # # -------------------------------------------------------------------------------------
        # # Solve deterministic ODE
        # ODE_sol = solve_ODE.solve_deterministic_ODE(param)
        # # Plot solution of deterministic model
        # plot.make_2D_plot(param, ODE_sol.t.flatten(), ODE_sol.y.flatten(), 'ODE_sol.png')
        # Plot stability functions
        compare_stability_functions.make_comparison(param)
        # # -------------------------------------------------------------------------------------
        # # Make bifurcation plots
        # # copy dataclass to prevent overwriting original
        # param_copy = dataclasses.replace(param)
        # param_copy.sol_directory_path = param.sol_directory_path
        # param_copy.stab_func_type = 'short_tail'
        # make_bifurcation_analysis.make_bifurcation_analysis(param_copy, data_domec)
        # param_copy.stab_func_type = 'long_tail'
        # make_bifurcation_analysis.make_bifurcation_analysis(param_copy, data_domec)
        # Plot potential
        # plot.plot_potentials(param)
    # -----------------------------------------------------------------------------------------
    # Run model with randomizations
    # Which parameters are parameterized is specified by the command line flags
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

    if stab_function_multi_noise:
        function_name = solve_SDE_stoch_stab_function.solve_SDE_with_stoch_stab_function_multi_noise
        sol_file_name = 'SDE_stab_func_multi_noise_sol'
        run_1D_SDE_model.solve_model_with_randomized_parameter(param, function_name, sol_file_name)

    if u_and_function:
        function_name = solve_SDEs.solve_SDE_with_stoch_u_and_internal_var
        sol_file_name = 'SDE_u_internal_var_sol'
        run_1D_SDE_model.solve_model_with_randomized_parameter(param, function_name, sol_file_name)

    if stab_function_multi_noise_u_td:
        function_name = solve_SDE_stoch_stab_function.solve_SDE_with_stoch_stab_function_multi_noise_time_dependent_u
        sol_file_name = 'SDE_stab_func_multi_noise_sol'
        run_1D_SDE_model.solve_model_with_randomized_parameter(param, function_name, sol_file_name)

    # -----------------------------------------------------------------------------------------
    # Save parameters
    save_parameters_in_file(param)


if __name__ == "__main__":
    # Read command line input
    f, sf, Q_i, Lam, wind, mp, sfmn, uf, sfu = parse_command_line_input.read_command_line_input()
    # -----------------------------------------------------------------------------------------
    # Load Parameters
    params = parameters.Parameters()
    # -----------------------------------------------------------------------------------------
    # Make directory for output
    params.sol_directory_path = 'output' + '/' + time.strftime("%Y%m%d_%H%M_%S") + '/'
    if not os.path.exists(params.sol_directory_path):
        os.makedirs(params.sol_directory_path)
        os.makedirs(params.sol_directory_path + 'temporary/')
    # -----------------------------------------------------------------------------------------
    # Run model and save output
    run_model(params, f, sf, Q_i, Lam, wind, mp, sfmn, uf, sfu)
