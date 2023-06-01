import os
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor

from one_D_model.utils import set_plotting_style

# ---------------------------------------------------------------------------
# Set plotting style and font sizes for figures
set_plotting_style.configure_plotting_style(figure_type='full_page_width')

# ---------------------------------------------------------------------------
# Define directory where simulation output is saved
output_directory = 'output/sensitivity_study/internal_variability/very_weakly/'
# output_directory = 'output/test_internal_var/'
# output_directory = 'output/test_internal_var_u/'

# Define name of solution file
# deltaT_file_name = '/SDE_u_sol_delta_T.npy'
deltaT_file_name = '/SDE_sol_delta_T.npy'
# deltaT_file_name = '/SDE_u_internal_var_sol_delta_T.npy'

# ---------------------------------------------------------------------------
# Define where the unstable equilibrium is located
location_unstable_eq = 12


# ---------------------------------------------------------------------------
def split_into_consecutive_ts(val, stepsize=1):
    array = np.split(val, np.where(np.diff(val) != stepsize)[0] + 1)
    if len(array) == 1:
        array = array[0]
    return array


def calculate_average_num_sim_with_transition(values):
    values = values[:,:,0]
    sim_with_trans = 0
    for idx, row in enumerate(values):
        if row[0] > location_unstable_eq:
            if any(row < location_unstable_eq):
                sim_with_trans += 1
        elif row[0] < location_unstable_eq:
            if any(row > location_unstable_eq):
                sim_with_trans += 1

    return sim_with_trans / np.shape(values)[0]


def combine_info_about_simulation_type(file_name):
    try:
        data = np.load(file_name)
    except OSError as e:
        return

    # Extract u value for simulation from file path
    path_elements = np.array(file_name.split('/'))
    idx_u_val = -3
    u_val = float(path_elements[idx_u_val].replace('_', '.'))

    # Extract sigma value(s) from file path
    if path_elements[-1] == 'SDE_sol_delta_T.npy':
        sigma_i = float(path_elements[idx_u_val + 1].replace('_', '.'))
        sigma_u = np.nan
    elif path_elements[-1] == 'SDE_u_sol_delta_T.npy':
        sigma_i = np.nan
        sigma_u = float(path_elements[idx_u_val + 1].replace('_', '.'))
    elif path_elements[-1] == 'SDE_u_internal_var_sol_delta_T.npy':
        sigma_i = float(path_elements[idx_u_val + 1].replace('_', '.'))
        sigma_u = float(path_elements[idx_u_val + 2].replace('_', '.'))

    # Calculate how many transitions take place on average over all simulations
    average_num_trans = calculate_average_num_sim_with_transition(data)

    # Save parameter combination for this simulation
    return (u_val, sigma_i, sigma_u, average_num_trans)

# Get all solution files
subdirectories = [x[0] for x in os.walk(output_directory)]
output_files = [subdir + deltaT_file_name for subdir in subdirectories]

parameter_comb = []

with ProcessPoolExecutor(max_workers=2) as executor:
    for result in executor.map(combine_info_about_simulation_type, output_files):
        if result:
            parameter_comb.append(result)

# Save calculated values in file
with open(output_directory+'average_transitions.json', 'w') as file:
    json.dump(parameter_comb, file)

# # Find minimal sigma(s) for every u for which at least x% of the simulations include a transition
# trans_percentage = 0.8
#
# u_range = np.unique([cur_u[0] for cur_u in parameter_comb])
#
# if deltaT_file_name != '/SDE_u_internal_var_sol_delta_T.npy':
#     first_sigma_with_enough_trans = []
#     for u in u_range:
#         # Find indices of the parameters which correspond to the current u
#         cor_idx = np.where([cur_u[0] for cur_u in parameter_comb] == u)[0]
#         # Find out how many transitions (on average) took place for the simulations corresponding to the current u
#         cor_average_num_trans = np.array([average_num_trans[i] for i in cor_idx])
#         idx_first_sigma_with_enough_trans = cor_idx[np.argmax(cor_average_num_trans >= trans_percentage)]
#         first_sigma_with_enough_trans.append(parameter_comb[idx_first_sigma_with_enough_trans][1])
#
#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     ax.scatter(u_range, first_sigma_with_enough_trans)
#     ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
#     plt.savefig(output_directory + 'transition_statistics.png', bbox_inches='tight', dpi=300)
