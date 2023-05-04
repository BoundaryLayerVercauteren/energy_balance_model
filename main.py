#!/usr/bin/env python
"""
Main script to run energy balance model (with perturbations).
"""
import os
import sys
import time
import json
import ast
import dataclasses
import numpy as np

# To be able to run this script on an external system
sys.path.append(os.getcwd())

from DomeC import process_dome_c_data
from one_D_model.utils import plot_output as plot, parse_command_line_input, set_plotting_style
from one_D_model.model import run_SDE_model as run_1D_SDE_model
from one_D_model.model import parameters, solve_ODE, make_bifurcation_analysis, solve_SDEs, \
    solve_SDE_stoch_stab_function, compare_stability_functions


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


def run_model(param, function=False, stab_function=False, Qi=False, Lambda=False, u=False, make_plot=False,
              u_and_function=False, stab_function_and_time_dependent_u=False, sensitivity_study=False):
    # If command line flag is given make plots
    if make_plot:
        # Set plotting style and font sizes for figures
        set_plotting_style.configure_plotting_style(figure_type='full_page_width')
        # Process Dome C data and plot it
        data_domec = process_dome_c_data.main()
        # -------------------------------------------------------------------------------------
        # Solve deterministic ODE
        ODE_sol = solve_ODE.solve_deterministic_ODE(param)
        # Plot solution of deterministic model
        plot.make_2D_plot(param, ODE_sol.t.flatten(), ODE_sol.y.flatten(), 'ODE_sol.png')
        # Plot stability functions
        compare_stability_functions.make_comparison(param.sol_directory_path)
        # -------------------------------------------------------------------------------------
        # Make bifurcation plots
        # copy dataclass to prevent overwriting original
        param_copy = dataclasses.replace(param)
        param_copy.sol_directory_path = param.sol_directory_path
        param_copy.stab_func_type = 'short_tail'
        make_bifurcation_analysis.make_bifurcation_analysis(param_copy, data_domec)
        param_copy.stab_func_type = 'long_tail'
        make_bifurcation_analysis.make_bifurcation_analysis(param_copy, data_domec)
        # -------------------------------------------------------------------------------------
        # Plot potential
        plot.plot_potentials(param)
    # -----------------------------------------------------------------------------------------
    # Run model with randomizations
    # Which parameters are parameterized is specified by the command line flags

    if sensitivity_study:
        u_step_size = 0.1
        u_range = np.round(np.arange(param.u_range_start, param.u_range_end + u_step_size, u_step_size), 3)
        sigma_step_size = 0.01
        sigma_range = np.round(np.arange(sigma_step_size, 1.0 + sigma_step_size, sigma_step_size), 3)
        orig_sol_directory_path = params.sol_directory_path
        for trans_type in ['weakly_very', 'very_weakly']:
            cur_sol_directory_path = orig_sol_directory_path + f'{trans_type}/'
            os.makedirs(cur_sol_directory_path)
            if trans_type == 'weakly_very':
                params.delta_T_0 = 4
            else:
                params.delta_T_0 = 24
            if function:
                for u_val in u_range:
                    for sigma_val in sigma_range:
                        params.sol_directory_path = cur_sol_directory_path + f"{str(u_val).replace('.', '_')}/{str(sigma_val).replace('.', '_')}/"
                        os.makedirs(params.sol_directory_path)
                        os.makedirs(params.sol_directory_path + 'temporary/')
                        param.U = u_val
                        param.sigma_delta_T = sigma_val
                        run_1D_SDE_model.solve_randomized_model(param)
                        # Save parameters
                        save_parameters_in_file(param)
            elif u:
                for u_val in u_range:
                    for sigma_val in sigma_range:
                        params.sol_directory_path = cur_sol_directory_path + f"{str(u_val).replace('.', '_')}/{str(sigma_val).replace('.', '_')}/"
                        os.makedirs(params.sol_directory_path)
                        os.makedirs(params.sol_directory_path + 'temporary/')
                        param.U = u_val
                        param.sigma_u = sigma_val
                        # Randomize wind velocity
                        function_name = solve_SDEs.solve_SDE_with_stoch_u
                        sol_file_name = 'SDE_u_sol'
                        run_1D_SDE_model.solve_model_with_randomized_parameter(param, function_name, sol_file_name)
                        # Save parameters
                        save_parameters_in_file(param)
            elif u_and_function:
                for u_val in u_range:
                    for sigma_deltaT_val in sigma_range:
                        for sigma_u_val in sigma_range:
                            params.sol_directory_path = cur_sol_directory_path + f"{str(u_val).replace('.', '_')}/{str(sigma_deltaT_val).replace('.', '_')}/{str(sigma_u_val).replace('.', '_')}/"
                            os.makedirs(params.sol_directory_path)
                            os.makedirs(params.sol_directory_path + 'temporary/')
                            param.U = u_val
                            param.sigma_u = sigma_u_val
                            param.sigma_delta_T = sigma_deltaT_val
                            # Randomize both wind velocity and the whole model
                            function_name = solve_SDEs.solve_SDE_with_stoch_u_and_internal_var
                            sol_file_name = 'SDE_u_internal_var_sol'
                            run_1D_SDE_model.solve_model_with_randomized_parameter(param, function_name, sol_file_name)
                            # Save parameters
                            save_parameters_in_file(param)
        return

    if function:
        # Randomize the whole model
        run_1D_SDE_model.solve_randomized_model(param)

    if Qi:
        # Randomize isothermal net radiation
        function_name = solve_SDEs.solve_SDE_with_stoch_Qi
        sol_file_name = 'SDE_Qi_sol'
        run_1D_SDE_model.solve_model_with_randomized_parameter(param, function_name, sol_file_name)

    if Lambda:
        # Randomize lumped parameter for feedbacks from both soil heat conduction and radiative cooling
        function_name = solve_SDEs.solve_SDE_with_stoch_lambda
        sol_file_name = 'SDE_lambda_sol'
        run_1D_SDE_model.solve_model_with_randomized_parameter(param, function_name, sol_file_name)

    if u:
        # Randomize wind velocity
        function_name = solve_SDEs.solve_SDE_with_stoch_u
        sol_file_name = 'SDE_u_sol'
        run_1D_SDE_model.solve_model_with_randomized_parameter(param, function_name, sol_file_name)

    if u_and_function:
        # Randomize both wind velocity and the whole model
        function_name = solve_SDEs.solve_SDE_with_stoch_u_and_internal_var
        sol_file_name = 'SDE_u_internal_var_sol'
        run_1D_SDE_model.solve_model_with_randomized_parameter(param, function_name, sol_file_name)

    if stab_function:
        # Randomize stability function
        function_name = solve_SDE_stoch_stab_function.solve_SDE_with_stoch_stab_function_multi_noise
        sol_file_name = 'SDE_stab_func_multi_noise_sol'
        run_1D_SDE_model.solve_model_with_randomized_parameter(param, function_name, sol_file_name)

    if stab_function_and_time_dependent_u:
        # Randomize stability function and make wind velocity time dependent rather than one fixed value
        # Calculate u range and save it
        param.u_range = create_u_range(param)
        np.savetxt(str(params.sol_directory_path) + 'u_range.txt', param.u_range)
        function_name = solve_SDE_stoch_stab_function.solve_SDE_with_stoch_stab_function_multi_noise_time_dependent_u
        sol_file_name = 'SDE_stab_func_multi_noise_sol'
        run_1D_SDE_model.solve_model_with_randomized_parameter(param, function_name, sol_file_name)

    # -----------------------------------------------------------------------------------------
    # Save parameters
    save_parameters_in_file(param)


if __name__ == "__main__":
    # Read command line input
    f, sf, Q_i, Lam, wind, mp, uf, sfu, ss = parse_command_line_input.read_command_line_input()
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
    run_model(params, f, sf, Q_i, Lam, wind, mp, uf, sfu, ss)
