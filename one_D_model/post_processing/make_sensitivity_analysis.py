import os
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor
import random

# ---------------------------------------------------------------------------
# Define directory where simulation output is saved
# output_directory = 'output/sensitivity_study/internal_variability/very_weakly/'
output_directory = '/uio/hypatia/geofag-personlig/metos-staff/amandink/01_energy_balance_model/01_output/sensitivity_study/internal_variability/weakly_very/'

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
    values = values[:, :, 0]
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

    # Take only a subset of the data
    if np.shape(data)[0] >= 1000:
        idx = random.sample(range(0, np.shape(data)[0]), 100)
        data = data[idx, :]

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
subdirectories = [directory for directory in subdirectories if directory[-1].isdigit()]
output_files = [subdir + deltaT_file_name for subdir in subdirectories]

parameter_comb = []

with ProcessPoolExecutor(max_workers=50) as executor:
    for result in executor.map(combine_info_about_simulation_type, output_files):
        if result:
            parameter_comb.append(result)

# Save calculated values in file
with open(output_directory + 'average_transitions.json', 'w') as file:
    json.dump(parameter_comb, file)
