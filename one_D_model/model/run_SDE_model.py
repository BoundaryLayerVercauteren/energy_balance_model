import multiprocessing
import os
from functools import partial

import numpy as np

from one_D_model.model import solve_SDEs


def solve_SDEs_wrapper(_, func_name, param):
    values = func_name(param)
    if np.shape(values)[1] > 1:
        return values[:, 0], values[:, 1]
    else:
        return values


def combine_npy_files(num_sim, sol_directory_path, file_string):
    solution = []
    path = sol_directory_path + 'temporary/'
    for idx in range(num_sim):
        curr_file = path + file_string + '_' + str(idx) + '.npy'
        data = np.load(curr_file)
        solution.append(data)
        os.remove(curr_file)
    np.save(sol_directory_path + file_string + '.npy', solution)


def solve_randomized_model(params):
    # Solve SDEs and save all solutions in a separate file
    with multiprocessing.Pool(processes=params.num_proc) as pool:
        for idx, res in enumerate(
                pool.imap_unordered(partial(solve_SDEs_wrapper, func_name=solve_SDEs.solve_SDE, param=params),
                                    range(params.num_simulation))):
            np.save(params.sol_directory_path + 'temporary/SDE_sol_delta_T_' + str(idx) + '.npy', res)

    # Combine solution files into a single one and delete single files
    combine_npy_files(params.num_simulation, params.sol_directory_path, 'SDE_sol_delta_T')


def solve_model_with_randomized_parameter(params, function_name, sol_file_name):
    with multiprocessing.Pool(processes=params.num_proc) as pool:
        for idx, res in enumerate(pool.imap_unordered(
                partial(solve_SDEs_wrapper, func_name=function_name, param=params),
                range(params.num_simulation))):
            np.save(params.sol_directory_path + 'temporary/' + sol_file_name + '_delta_T_' + str(idx) + '.npy',
                    res[0][:])
            np.save(params.sol_directory_path + 'temporary/' + sol_file_name + '_param_' + str(idx) + '.npy', res[1][:])

    # Combine solution files into a single one and delete single files
    combine_npy_files(params.num_simulation, params.sol_directory_path, sol_file_name + '_delta_T')
    combine_npy_files(params.num_simulation, params.sol_directory_path, sol_file_name + '_param')


# Very slow!
def solve_model_with_randomized_parameter_z0(params):
    SDE_z0_sol = []
    with multiprocessing.Pool(processes=params.num_proc) as pool:
        for res_z0 in pool.imap_unordered(
                partial(solve_SDEs.solve_SDE_with_stoch_z0, param=params), range(params.num_simulation)):
            SDE_z0_sol.append(res_z0)

    SDE_z0_sol_delta_T = np.array([SDE_z0_sol[idx][1] for idx in range(params.num_simulation)])
    SDE_z0_sol_time = np.array([SDE_z0_sol[idx][0] for idx in range(params.num_simulation)])

    SDE_z0_sol_z0 = np.zeros((np.shape(SDE_z0_sol)[0], np.shape(SDE_z0_sol_time)[1]))
    SDE_z0_sol_z0[:] = np.nan

    for sim_idx in range(np.shape(SDE_z0_sol)[0]):
        for time_idx, time in enumerate(SDE_z0_sol_time[sim_idx, :]):
            for elem in SDE_z0_sol[sim_idx][2]:
                if np.around(elem[0], 0) == np.around(time, 0):
                    SDE_z0_sol_z0[sim_idx, time_idx] = elem[1]
    print(SDE_z0_sol_z0[0, :])
    np.save(params.sol_directory_path + 'SDE_z0_sol_delta_T.npy', SDE_z0_sol_delta_T)
    np.save(params.sol_directory_path + 'SDE_z0_sol_param.npy', SDE_z0_sol_z0)
    np.save(params.sol_directory_path + 'SDE_z0_sol_time.npy', SDE_z0_sol_time)
