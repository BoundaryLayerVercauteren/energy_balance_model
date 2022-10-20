import multiprocessing
from functools import partial

import numpy as np

from one_D_model.model import solve_SDEs, solve_SDE_stoch_stab_function


def solve_SDEs_wrapper(_, func_name, param):
    values = func_name(param)
    if np.shape(values)[1] > 1:
        return values[:, 0], values[:, 1]
    else:
        return values


SDE_u_sol = []
SDE_delta_T_sol = []
SDE_Qi_sol = []
SDE_lambda_sol = []
SDE_stab_func_sol = []
SDE_z0_sol = []


def main(params):
    # Solve SDEs and run Monte Carlo Simulation
    num_simulation = params.num_simulation
    # ------------------------------------------------------------
    with multiprocessing.Pool(processes=params.num_proc) as pool:
        for res in pool.imap_unordered(partial(solve_SDEs_wrapper, func_name=solve_SDEs.solve_SDE, param=params), range(num_simulation)):
            SDE_delta_T_sol.append(res)

    np.save(params.sol_directory_path + 'SDE_sol_delta_T.npy', SDE_delta_T_sol)
    # ------------------------------------------------------------
    with multiprocessing.Pool(processes=params.num_proc) as pool:

        for res_u in pool.imap_unordered(partial(solve_SDEs_wrapper, func_name=solve_SDEs.solve_SDE_with_stoch_u, param=params), range(num_simulation)):
            SDE_u_sol.append(res_u)

    SDE_u_sol_delta_T = np.array([SDE_u_sol[idx][0][:] for idx in range(params.num_simulation)])
    SDE_u_sol_u = np.array([SDE_u_sol[idx][1][:] for idx in range(params.num_simulation)])

    np.save(params.sol_directory_path + 'SDE_u_sol_delta_T.npy', SDE_u_sol_delta_T)
    np.save(params.sol_directory_path + 'SDE_u_sol_u.npy', SDE_u_sol_u)
    # ------------------------------------------------------------
    with multiprocessing.Pool(processes=params.num_proc) as pool:
        for res_Qi in pool.imap_unordered(partial(solve_SDEs_wrapper, func_name=solve_SDEs.solve_SDE_with_stoch_Qi, param=params), range(num_simulation)):
            SDE_Qi_sol.append(res_Qi)

    SDE_Qi_sol_delta_T = np.array([SDE_Qi_sol[idx][0][:] for idx in range(params.num_simulation)])
    SDE_Qi_sol_Qi = np.array([SDE_Qi_sol[idx][1][:] for idx in range(params.num_simulation)])

    np.save(params.sol_directory_path + 'SDE_Qi_sol_delta_T.npy', SDE_Qi_sol_delta_T)
    np.save(params.sol_directory_path + 'SDE_Qi_sol_Qi.npy', SDE_Qi_sol_Qi)
    # ------------------------------------------------------------
    with multiprocessing.Pool(processes=params.num_proc) as pool:
        for res_lambda in pool.imap_unordered(partial(solve_SDEs_wrapper, func_name=solve_SDEs.solve_SDE_with_stoch_lambda, param=params), range(num_simulation)):
            SDE_lambda_sol.append(res_lambda)

    SDE_lambda_sol_delta_T = np.array([SDE_lambda_sol[idx][0][:] for idx in range(params.num_simulation)])
    SDE_lambda_sol_lambda = np.array([SDE_lambda_sol[idx][1][:] for idx in range(params.num_simulation)])

    np.save(params.sol_directory_path + 'SDE_lambda_sol_delta_T.npy', SDE_lambda_sol_delta_T)
    np.save(params.sol_directory_path + 'SDE_lambda_sol_lambda.npy', SDE_lambda_sol_lambda)
    # ------------------------------------------------------------
    with multiprocessing.Pool(processes=params.num_proc) as pool:
        for res_sf in pool.imap_unordered(
                partial(solve_SDEs_wrapper, func_name=solve_SDE_stoch_stab_function.solve_SDE_with_stoch_stab_function, param=params),
                range(num_simulation)):
            SDE_stab_func_sol.append(res_sf)

    SDE_stab_func_sol_delta_T = np.array([SDE_stab_func_sol[idx][0][:] for idx in range(params.num_simulation)])
    SDE_stab_func_sol_sf = np.array([SDE_stab_func_sol[idx][1][:] for idx in range(params.num_simulation)])

    np.save(params.sol_directory_path + 'SDE_stab_func_sol_delta_T.npy', SDE_stab_func_sol_delta_T)
    np.save(params.sol_directory_path + 'SDE_stab_func_sol_sf.npy', SDE_stab_func_sol_sf)
    # ------------------------------------------------------------
    with multiprocessing.Pool(processes=params.num_proc) as pool:
        for res_z0 in pool.imap_unordered(
                partial(solve_SDEs.solve_SDE_with_stoch_z0, param=params), range(num_simulation)):
            SDE_z0_sol.append(res_z0)

    SDE_z0_sol_delta_T = np.array([SDE_z0_sol[idx][1] for idx in range(params.num_simulation)])
    SDE_z0_sol_time = np.array([SDE_z0_sol[idx][0] for idx in range(params.num_simulation)])

    SDE_z0_sol_z0 = np.zeros((np.shape(SDE_z0_sol)[0], np.shape(SDE_z0_sol_time)[1]))

    for sim_idx in range(np.shape(SDE_z0_sol)[0]):
        for time_idx, time in enumerate(SDE_z0_sol_time[sim_idx, :]):
            for elem in SDE_z0_sol[sim_idx][2]:
                if elem['time'] == time:
                    SDE_z0_sol_z0[sim_idx, time_idx] = elem['z0']

    np.save(params.sol_directory_path + 'SDE_z0_sol_delta_T.npy', SDE_z0_sol_delta_T)
    np.save(params.sol_directory_path + 'SDE_z0_sol_z0.npy', SDE_z0_sol_z0)
    np.save(params.sol_directory_path + 'SDE_z0_sol_time.npy', SDE_z0_sol_time)
