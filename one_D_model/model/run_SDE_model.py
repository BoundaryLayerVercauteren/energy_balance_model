import multiprocessing
from functools import partial

import numpy as np

from one_D_model.model import solve_SDEs


def solve_SDEs_wrapper(func_name, param, sim_idx):
    values = func_name(param)
    if np.shape(values)[1] > 1:
        return values[:, 0], values[:, 1]
    else:
        return values


def main(params):
    # Solve SDEs and run Monte Carlo Simulation
    num_simulation = params.num_simulation

    with multiprocessing.Pool(processes=params.num_proc) as pool:
        SDE_delta_T_sol = pool.map(partial(solve_SDEs_wrapper, solve_SDEs.solve_SDE, params), range(num_simulation))
        SDE_u_sol = pool.map(partial(solve_SDEs_wrapper, solve_SDEs.solve_SDE_with_stoch_u, params),
                             range(num_simulation))
        SDE_Qi_sol = pool.map(partial(solve_SDEs_wrapper, solve_SDEs.solve_SDE_with_stoch_Qi, params),
                              range(num_simulation))
        SDE_lambda_sol = pool.map(partial(solve_SDEs_wrapper, solve_SDEs.solve_SDE_with_stoch_lambda, params),
                                  range(num_simulation))

    # Save simulation output
    SDE_u_sol_delta_T = np.array([SDE_u_sol[idx][0][:] for idx in range(params.num_simulation)])
    SDE_u_sol_u = np.array([SDE_u_sol[idx][1][:] for idx in range(params.num_simulation)])
    SDE_Qi_sol_delta_T = np.array([SDE_Qi_sol[idx][0][:] for idx in range(params.num_simulation)])
    SDE_Qi_sol_Qi = np.array([SDE_Qi_sol[idx][1][:] for idx in range(params.num_simulation)])
    SDE_lambda_sol_delta_T = np.array([SDE_lambda_sol[idx][0][:] for idx in range(params.num_simulation)])
    SDE_lambda_sol_lambda = np.array([SDE_lambda_sol[idx][1][:] for idx in range(params.num_simulation)])

    np.save(params.sol_directory_path + 'SDE_sol_delta_T.npy', SDE_delta_T_sol)

    np.save(params.sol_directory_path + 'SDE_u_sol_delta_T.npy', SDE_u_sol_delta_T)
    np.save(params.sol_directory_path + 'SDE_u_sol_u.npy', SDE_u_sol_u)

    np.save(params.sol_directory_path + 'SDE_Qi_sol_delta_T.npy', SDE_Qi_sol_delta_T)
    np.save(params.sol_directory_path + 'SDE_Qi_sol_Qi.npy', SDE_Qi_sol_Qi)

    np.save(params.sol_directory_path + 'SDE_lambda_sol_delta_T.npy', SDE_lambda_sol_delta_T)
    np.save(params.sol_directory_path + 'SDE_lambda_sol_lambda.npy', SDE_lambda_sol_lambda)

