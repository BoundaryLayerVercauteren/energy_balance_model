import numpy as np
import random
import os

# Define directory where data is saved
data_directory = 'output/1000_sim_short_tail_stab_func_multi_noise_var_u/sigma_0_1_start_very/simulations/'
file_name_1 = '/SDE_stab_func_multi_noise_sol_delta_T.npy'
file_name_2 = '/SDE_stab_func_multi_noise_sol_param.npy'

subset_size = 500


def reduce_num_simulation(file_path_1, file_path_2):
    try:
        data_1 = np.load(file_path_1)
        data_2 = np.load(file_path_2)
    except OSError as e:
        return

    # Save subset of the data
    if np.shape(data_1)[0] > subset_size:
        idx = random.sample(range(0, np.shape(data_1)[0]), subset_size)
        data_1 = data_1[idx, :]
        data_2 = data_2[idx, :]
        np.save(file_path_1, data_1)
        np.save(file_path_2, data_2)
    else:
        return

reduce_num_simulation(data_directory+file_name_1, data_directory+file_name_2)

# # Get all solution files
# subdirectories = [x[0] for x in os.walk(data_directory)]
# subdirectories = [directory for directory in subdirectories if directory[-1].isdigit()]
# all_file_paths = [subdir + file_name for subdir in subdirectories]
#
# for loop_file_path in all_file_paths:
#     reduce_num_simulation(loop_file_path)

print('done')
