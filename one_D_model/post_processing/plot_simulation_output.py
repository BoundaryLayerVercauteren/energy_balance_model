"""Needs to be run with: python -m one_D_model.post_processing.plot_simulation_output"""
import dataclasses
import os
from dataclasses import dataclass

import cmcrameri as cmc
import matplotlib
import numpy as np
from dataclasses_json import dataclass_json
from matplotlib import pyplot as plt

from one_D_model.model import solve_ODE
from one_D_model.utils import plot_output

# Define directory where simulation output is saved
output_directory = 'output/20230719_1159_24/'

# Specify what type of simulation was run and shall be plotted
sim_type_options = ['internal_var', 'wind', 'internal_var_wind', 'turbulence']
sim_type = sim_type_options[3]

# Load data
if sim_type == sim_type_options[0]:
    file_name_delta_T = 'SDE_internal_var_sol_delta_T'
    SDE_sol_delta_T = np.load(output_directory + file_name_delta_T + '.npy')
else:
    if sim_type == sim_type_options[1]:
        file_name_delta_T = 'SDE_u_sol_delta_T'
        file_name_param = 'SDE_u_sol_param'
    elif sim_type == sim_type_options[2]:
        file_name_delta_T = 'SDE_u_internal_var_sol_delta_T'
        file_name_param = 'SDE_u_internal_var_sol_param'
    elif sim_type == sim_type_options[3]:
        file_name_delta_T = 'SDE_stab_func_sol_delta_T'
        file_name_param = 'SDE_stab_func_sol_param'
    SDE_sol_delta_T = np.load(output_directory + file_name_delta_T + '.npy')
    SDE_sol_param = np.load(output_directory + file_name_param + '.npy')


# Load parameters
@dataclass_json
@dataclass
class Parameters:
    t_start: float
    t_end_h: float
    t_end: float
    dt: float
    num_steps: float
    num_simulation: int
    stab_func_type: str
    Lambda: float
    Q_i: float
    z0: float
    zr: float
    grav: float
    Tr: float
    alpha: float
    kappa: float
    cv: float
    rho: float
    cp: float


with open(output_directory + 'parameters.json', 'r') as file:
    param_data = file.read()

params = Parameters.from_json(param_data)

params.t_span_h = np.linspace(params.t_start, params.t_end_h, params.num_steps)

# Make directory for visualizations
params.sol_directory_path = output_directory + 'visualizations/'
if not os.path.exists(params.sol_directory_path):
    os.makedirs(params.sol_directory_path)


def make_distribution_plot(values, params, file_name, xlabel):
    fig = plt.figure(figsize=(5, 5))

    color = matplotlib.cm.get_cmap('cmc.batlow', 1).colors

    plt.axvline(x=24, color='r')
    plt.axvline(x=4, color='r')
    plt.axvline(x=12, color='r', linestyle='--')
    plt.hist(values, 100, color=color[0])
    #plt.ylim((0, 3.25 * 10 ** 6))
    plt.xlabel(xlabel)
    plt.ylabel(r'Density of $\Delta T$')
    plt.savefig(params.sol_directory_path + file_name, bbox_inches='tight', dpi=300)
    # To clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close('all')  # Closes all the figure windows.


# Plot distribution of Delta T solution
make_distribution_plot(SDE_sol_delta_T.flatten(), params, 'distribution_' + file_name_delta_T,
                       r'$\Delta T$ [K]')


# Overlay potential plot with distribution plot
def plot_potentials_and_output_distribution(param_class, delta_T_data, file_str):
    delta_T_range = np.arange(0, 30, 0.5)
    u_list_st = [5.3, 5.6, 5.9]
    u_list_lt = [4.87, 4.89, 4.9]

    color = matplotlib.cm.get_cmap('cmc.batlow', 4).colors
    markers = ['s', 'p', 'v']

    # copy dataclass to prevent overwriting original
    param_copy = dataclasses.replace(param_class)

    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    fig.subplots_adjust(right=0.75)

    ax2 = ax1.twinx()

    if param_copy.stab_func_type == 'short_tail':
        p0 = ax2.hist(delta_T_data, 100, color=color[1], alpha=0.5, zorder=0)
        for idx, u_elem in enumerate(u_list_st):
            potential_st = solve_ODE.calculate_potential(delta_T_range, u_elem, param_copy)
            p1, = ax1.plot(delta_T_range, - potential_st, label='u = ' + str(u_elem), color=color[idx],
                           marker=markers[idx], markevery=5, zorder=10)
        title = 'a)'

    elif param_copy.stab_func_type == 'long_tail':
        p2 = ax2.hist(delta_T_data, 100, color=color[1], alpha=0.5, zorder=0)
        for idx, u_elem in enumerate(u_list_lt):
            potential_lt = solve_ODE.calculate_potential(delta_T_range, u_elem, param_copy)
            p3, = ax1.plot(delta_T_range, - potential_lt, label='u = ' + str(u_elem), color=color[idx],
                           marker=markers[idx], markevery=5, zorder=10)
        title = 'b)'

    ax1.set_xlabel(r'$\Delta T$ [K]')
    ax1.set_ylabel('V [$K^2$/s]')
    ax2.set_ylabel(r'Density of $\Delta T$', color=color[1], alpha=1)
    ax2.tick_params(axis="y", labelcolor=color[1])

    #ax2.set_ylim((0.0, 4.5 * 10 ** 6))
    ax1.set_title(title, loc='left')
    ax1.legend()  # loc='upper left')

    plt.savefig(f'{param_class.sol_directory_path}{file_str}_dist_potentials_{param_copy.stab_func_type}.pdf',
                bbox_inches='tight', dpi=300)


plot_potentials_and_output_distribution(params, SDE_sol_delta_T.flatten(), file_name_delta_T)

# Plot ten time series for every simulation type
if params.num_simulation >= 1:
    vis_idx = np.linspace(0, params.num_simulation - 1, 10).astype(int)
    for idx in vis_idx:
        if sim_type == sim_type_options[0]:
            plot_output.make_line_plot_of_single_solution(params, params.t_span_h, SDE_sol_delta_T[idx][:].flatten(),
                                                          'SDE_sol_delta_T_over_time_sim' + str(
                                                              idx) + '.pdf', xlabel='t [h]',
                                                          ylabel=r'$\Delta T$ [K]')
        else:
            if sim_type == sim_type_options[1] or sim_type == sim_type_options[2]:
                line_labels = [r'$\Delta T$', 'U']
                y_axes_label_1 = r'$\Delta T$ [K]'
                y_axes_label_2 = r'U [$\mathrm{ms^{-1}}$]'
            elif sim_type == sim_type_options[3]:
                line_labels = [r'$\Delta T$', r'$\phi$']
                y_axes_label_1 = r'$\Delta T$ [K]'
                y_axes_label_2 = r'$\phi$'

            plot_output.make_line_plot_of_solution_and_parameter(params, params.t_span_h,
                                                                 np.array([SDE_sol_delta_T[idx, :],
                                                                           SDE_sol_param[idx, :]]).T,
                                                                 line_labels,
                                                                 f'{file_name_delta_T}_over_time_sim{idx}.pdf',
                                                                 xlabel='t [h]', ylabel=y_axes_label_1,
                                                                 ylabel2=y_axes_label_2)
