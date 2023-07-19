#!/usr/bin/env python
"""
Main script to run conceptual model for temperature inversions (with and without perturbations).
"""
import ast
import dataclasses
import json
import os
import sys
import time

import numpy as np

# To be able to run this script on an external system
sys.path.append(os.getcwd())

from DomeC import process_dome_c_data
from one_D_model.utils import plot_output, parse_command_line_input, set_plotting_style
from one_D_model.model import run_SDE_model, parameters, solve_ODE, make_bifurcation_analysis, solve_SDEs, \
    compare_stability_functions


def save_parameters_in_file(param_vals):
    # Define name of parameter file
    file_name = str(param_vals.sol_directory_path) + 'parameters.json'
    # Transform parameter class to json
    params_json = param_vals.to_json()
    # Save json to file + remove unnecessary characters
    with open(file_name, 'w') as file:
        json.dump(ast.literal_eval(params_json), file)


def create_u_range(param_vals, num_u_steps=50):
    # Define which values u can take
    u_values = np.round(np.linspace(param_vals.u_range_start, param_vals.u_range_end, num_u_steps), 1)
    # Calculate how often each u value will be repeated
    num_repeat = int(param_vals.num_steps / num_u_steps)
    # Define u_range
    u = np.array([np.repeat(u_values[idx], num_repeat) for idx in np.arange(0, num_u_steps)])
    return u.flatten()


def make_ode_plots(param):
    """Create all plots which are relevant to study the ODE model (eq. 2)."""
    # Set plotting style and font sizes for figures
    set_plotting_style.configure_plotting_style(figure_type='full_page_width')

    # Process Dome C data
    data_domec = process_dome_c_data.main()

    # Solve ODE
    ODE_sol = solve_ODE.solve_ODE(param)

    # Plot solution of ODE
    plot_output.make_line_plot_of_single_solution(param, ODE_sol.t.flatten(), ODE_sol.y.flatten(), 'ODE_sol.png')

    # Plot stability functions (figure 2)
    compare_stability_functions.make_comparison(param.sol_directory_path)

    # Make bifurcation plots (figure 1)
    # copy dataclass to prevent overwriting original
    param_copy = dataclasses.replace(param)
    param_copy.sol_directory_path = param.sol_directory_path
    param_copy.stab_func_type = 'short_tail'
    make_bifurcation_analysis.make_bifurcation_analysis(param_copy, data_domec, save_values=True)
    param_copy.stab_func_type = 'long_tail'
    make_bifurcation_analysis.make_bifurcation_analysis(param_copy, data_domec)

    # Plot potential (figure 3)
    plot_output.plot_potentials(param)


def solve_ode_with_variable_u(param):
    """Solve ODE model (eq. 2) with variable wind. This function was used to create the data for the orange line in
     figure 10."""
    param.u_range = create_u_range(param)
    if param.delta_T_0 < 12:
        param.u_range = param.u_range[::-1]
    np.savetxt(param.sol_directory_path + 'u_range.txt', param.u_range)
    ode_sol = solve_ODE.solve_ODE_with_time_dependent_u(param)
    np.save(param.sol_directory_path + 'ODE_sol_delta_T.npy', ode_sol)


def run_sde_model(param, **randomization_type):
    """Run conceptual model for temperature inversions with different types of randomizations. See section 2.3 and its
     subsections."""
    if randomization_type.get('function'):
        # Solve model with small-scale fluctuations of an unresolved process (eq. 3)
        run_SDE_model.solve_randomized_model(param)
    if randomization_type.get('wind'):
        # Solve model with small-scale fluctuations of the wind forcing (eq. 4)
        run_SDE_model.solve_model_with_randomized_parameter(param, solve_SDEs.solve_SDE_with_stoch_u, 'SDE_u_sol')
    if randomization_type.get('wind_and_function'):
        # Solve model with the previous two randomizations combined (eq. 5)
        run_SDE_model.solve_model_with_randomized_parameter(param,
                                                            solve_SDEs.solve_SDE_with_stoch_u_and_internal_var,
                                                            'SDE_u_internal_var_sol')
    if randomization_type.get('stab_function'):
        # Solve model with a stochastic stability function (eq. 6)
        run_SDE_model.solve_model_with_randomized_parameter(param, solve_SDEs.solve_SDE_with_stoch_stab_function,
                                                            'SDE_stab_func_sol')
    # Save parameters
    save_parameters_in_file(param)


def run_sde_model_with_nonconstant_wind(param, **randomization_type):
    """Solve SDE model with stochastic stability function (eq. 6) and variable wind. See section 2.3.3. and figure 10
    for a detailed description."""
    param.u_range = create_u_range(param)
    if param.delta_T_0 < 12:
        param.u_range = param.u_range[::-1]
    np.savetxt(str(param.sol_directory_path) + 'u_range.txt', param.u_range)

    if randomization_type.get('stab_function'):
        run_SDE_model.solve_model_with_randomized_parameter(param,
                                                            solve_SDEs.solve_SDE_with_stoch_stab_function_time_dependent_u,
                                                            'SDE_stab_func_sol')


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path + 'temporary/')


def make_sensitivity_study_for_sde_model(param, **randomization_type):
    """Perform sensitivity analysis for one of the randomized models (eq. 3-6). See for example section 2.3.1. and
    figure 4 for a detailed description."""
    u_step_size = 0.1
    u_range = np.round(np.arange(param.u_range_start, param.u_range_end + u_step_size, u_step_size), 3)
    sigma_step_size = 0.02
    sigma_range = np.round(np.arange(sigma_step_size, .5 + sigma_step_size, sigma_step_size), 3)
    orig_sol_directory_path = params.sol_directory_path
    for trans_type in ['weakly_very', 'very_weakly']:
        cur_sol_directory_path = orig_sol_directory_path + f'{trans_type}/'
        os.makedirs(cur_sol_directory_path)
        if trans_type == 'weakly_very':
            params.delta_T_0 = 4
        else:
            params.delta_T_0 = 24
        if randomization_type.get('function'):
            for u_val in u_range:
                for sigma_val in sigma_range:
                    params.sol_directory_path = cur_sol_directory_path + f"{str(u_val).replace('.', '_')}/{str(sigma_val).replace('.', '_')}/"
                    create_directory(params.sol_directory_path)
                    param.U = u_val
                    param.sigma_delta_T = sigma_val
                    run_sde_model(param, function=True)
        if randomization_type.get('stab_function'):
            for u_val in u_range:
                for sigma_val in sigma_range:
                    params.sol_directory_path = cur_sol_directory_path + f"{str(u_val).replace('.', '_')}/" \
                                                                         f"{str(sigma_val).replace('.', '_')}/"
                    create_directory(params.sol_directory_path)
                    param.U = u_val
                    param.sigma_phi = sigma_val
                    run_sde_model(param, stab_function=True)
        if randomization_type.get('wind_and_function'):
            sigma_u_range = np.array([0.01, 0.04])
            sigma_range = np.insert(sigma_range, 0, 0, axis=0)
            for u_val in u_range:
                for sigma_deltaT_val in sigma_range:
                    for sigma_u_val in sigma_u_range:
                        params.sol_directory_path = cur_sol_directory_path + f"{str(u_val).replace('.', '_')}/" \
                                                                             f"{str(sigma_deltaT_val).replace('.', '_')}" \
                                                                             f"/{str(sigma_u_val).replace('.', '_')}/"
                        create_directory(params.sol_directory_path)
                        param.U = u_val
                        param.sigma_u = sigma_u_val
                        param.sigma_delta_T = sigma_deltaT_val
                        run_sde_model(param, wind_and_function=True)


if __name__ == "__main__":
    # Read command line input
    f, sf, u, pl, uf, sfu, ss, odeu = parse_command_line_input.read_command_line_input()
    # -----------------------------------------------------------------------------------------
    # Load Parameters
    params = parameters.Parameters()
    # -----------------------------------------------------------------------------------------
    # Make directory for output
    params.sol_directory_path = 'output' + '/' + time.strftime("%Y%m%d_%H%M_%S") + '/'
    create_directory(params.sol_directory_path)
    # -----------------------------------------------------------------------------------------
    # Create plots for ODE model
    if pl:
        make_ode_plots(params)
    # Solve ODE with variable u
    if odeu:
        solve_ode_with_variable_u(params)
    # -----------------------------------------------------------------------------------------
    # Run sensitivity analysis for SDE model
    if ss:
        make_sensitivity_study_for_sde_model(params, function=f, wind_and_function=uf, stab_function=sf)
    # Run SDE model (for one fixed wind velocity)
    else:
        run_sde_model(params, function=f, wind=u, wind_and_function=uf, stab_function=sf)
    # Run SDE model with time dependent wind velocity
    if sfu:
        run_sde_model_with_nonconstant_wind(params, stab_function=sfu)
