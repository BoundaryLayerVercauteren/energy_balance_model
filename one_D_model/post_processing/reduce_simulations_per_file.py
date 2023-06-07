import numpy as np
import random
import os

# Define directory where data is saved
data_directory = 'output/sensitivity_study/internal_variability/'
file_name = '/SDE_sol_delta_T.npy'

subset_size = 500


def reduce_num_simulation(file_path):
    try:
        data = np.load(file_path)
    except OSError as e:
        return

    # Save subset of the data
    if np.shape(data)[0] > subset_size:
        idx = random.sample(range(0, np.shape(data)[0]), subset_size)
        data = data[idx, :]
        np.save(file_path, data)
    else:
        return


# Get all solution files
subdirectories = [x[0] for x in os.walk(data_directory)]
subdirectories = [directory for directory in subdirectories if directory[-1].isdigit()]
all_file_paths = [subdir + file_name for subdir in subdirectories]

for loop_file_path in all_file_paths:
    reduce_num_simulation(loop_file_path)

print('done')
