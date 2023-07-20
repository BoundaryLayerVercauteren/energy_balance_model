"""Script to perform Monte Carlo simulation of SDEs (eq. 3-6)."""

import multiprocessing
import os
from functools import partial

import numpy as np

from one_D_model.model import solve_SDEs


def solve_SDEs_wrapper(_, func_name, param):
    """Wrapper function to transform SDE results into desired shape."""
    values = func_name(param)
    if np.shape(values)[1] > 1:
        return values[:, 0], values[:, 1]
    else:
        return values


def combine_npy_files(num_sim, sol_directory_path, file_string):
    """Combine all data files in the directory temporary to one and delete them."""
    solution = []
    path = sol_directory_path + "temporary/"
    for idx in range(num_sim):
        curr_file = path + file_string + "_" + str(idx) + ".npy"
        data = np.load(curr_file)
        solution.append(data)
        os.remove(curr_file)
    np.save(sol_directory_path + file_string + ".npy", solution)


def solve_randomized_model(params):
    """Run Monte Carlo simulation of conceptual model for temperature inversions with additive noise term to represent
    small-scale fluctuations of an unresolved process (eq. 3). The simulations are run in parallel.
    """
    # Solve SDEs and save all solutions in a separate file
    with multiprocessing.Pool(processes=params.num_proc) as pool:
        for idx, res in enumerate(
            pool.imap_unordered(
                partial(
                    solve_SDEs_wrapper,
                    func_name=solve_SDEs.solve_SDE_with_internal_var,
                    param=params,
                ),
                range(params.num_simulation),
            )
        ):
            np.save(
                params.sol_directory_path
                + "temporary/SDE_internal_var_sol_delta_T_"
                + str(idx)
                + ".npy",
                res,
            )

    # Combine solution files into a single one and delete single files
    combine_npy_files(
        params.num_simulation, params.sol_directory_path, "SDE_internal_var_sol_delta_T"
    )


def solve_model_with_randomized_parameter(params, function_name, sol_file_name):
    """Run Monte Carlo simulation of perturbed conceptual model for temperature inversions (eq. 4, 5, 6). The
    simulations are run in parallel."""
    with multiprocessing.Pool(processes=params.num_proc) as pool:
        for idx, res in enumerate(
            pool.imap_unordered(
                partial(solve_SDEs_wrapper, func_name=function_name, param=params),
                range(params.num_simulation),
            )
        ):
            if np.shape(res)[1] > 1:
                np.save(
                    params.sol_directory_path
                    + "temporary/"
                    + sol_file_name
                    + "_delta_T_"
                    + str(idx)
                    + ".npy",
                    res[0][:],
                )
                np.save(
                    params.sol_directory_path
                    + "temporary/"
                    + sol_file_name
                    + "_param_"
                    + str(idx)
                    + ".npy",
                    res[1][:],
                )
            else:
                np.save(
                    params.sol_directory_path
                    + "temporary/"
                    + sol_file_name
                    + "_delta_T_"
                    + str(idx)
                    + ".npy",
                    res.flatten(),
                )

    # Combine solution files into a single one and delete single files
    combine_npy_files(
        params.num_simulation, params.sol_directory_path, sol_file_name + "_delta_T"
    )
    if np.shape(res)[1] > 1:
        combine_npy_files(
            params.num_simulation, params.sol_directory_path, sol_file_name + "_param"
        )
