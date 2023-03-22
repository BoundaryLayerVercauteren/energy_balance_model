import numpy as np
import sdeint

from one_D_model.model import solve_ODE
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt


# First define the scaling functions that scale the SDE parameter values with Ri
def kappa(x):
    # Coefficients of the Kappa function
    aK = 9.3212
    bK = 0.9088
    cK = 0.0738
    dK = 8.3220
    Kappa = aK * np.tanh(bK * np.log10(x) - cK) + dK
    return Kappa


def upsilon(x):
    # Coefficients of the Upsilon function
    aU = 0.4294
    bU = 0.1749
    Upsilon = 10 ** (aU * np.log10(x) + bU)
    return Upsilon


def sigma(x, sigma_s):
    # Coefficients of the Sigma function
    aS = 0.8069
    bS = 0.6044
    cS = 0.8368
    dS = sigma_s
    Sigma = 10 ** (aS * np.tanh(bS * np.log10(x) - cS) + dS)
    return Sigma


def define_SDE(t, delta_T, phi_stoch, param):
    f_stab = 1 / phi_stoch
    c_D = (param.kappa / np.math.log(param.zr / param.z0)) ** 2
    return (1 / param.cv) * (
            param.Q_i - param.Lambda * delta_T - param.rho * param.cp * c_D * param.U * delta_T * f_stab)


def define_stoch_stab_function(delta_T, phi_stoch, t, param):
    Ri = solve_ODE.calculate_richardson_number(param, delta_T, param.U)
    return (1 / 3600) * (1 + kappa(Ri) * phi_stoch - upsilon(Ri) * phi_stoch ** 2)


def solve_SDE_with_stoch_stab_function(param):
    """Original model by van de Wiel with stochastic stability function given by Boyko et. al, 2023."""
    # Combine initial conditions for
    phi_0 = 1 / solve_ODE.calculate_stability_function(param, param.delta_T_0, param.U)
    initial_cond = np.array([param.delta_T_0, phi_0])

    # Define functions for 2D SDE
    def _f(X, t):
        return np.array([define_SDE(t, X[0], X[1], param), define_stoch_stab_function(X[0], X[1], t, param)])

    def _G(X, t):
        Ri = solve_ODE.calculate_richardson_number(param, param.delta_T_0, param.U)
        return np.diag([0.0, (1 / np.sqrt(3600)) * (sigma(Ri, param.sigma_s) * X[1])])

    result = sdeint.itoint(_f, _G, initial_cond, param.t_span)

    # phi = 1/ regular stability function
    result[:, 1] = 1 / result[:, 1]

    return result

#
# def define_poisson_stab_function(params, delta_T_val, U_val, critical_Ri=0.25):
#     np.random.seed()
#     # Define Poisson random variable
#     poisson_rv = np.random.poisson(lam=params.poisson_lambda, size=1) * 0.1
#     # Calculate Richardson number
#     Ri_val = solve_ODE.calculate_richardson_number(params, delta_T_val, U_val)
#     # Calculate stochastic stability function for given Richardson
#     stab_function = solve_ODE.calculate_stability_function(params, delta_T_val, U_val)
#     # if Ri_val <= critical_Ri:
#     #     stab_function = solve_ODE.calculate_stability_function(params, delta_T_val, U_val)
#     # else:
#     #     stab_function = solve_ODE.calculate_stability_function(params, delta_T_val, U_val) + poisson_rv
#     # if delta_T_val > 30:
#     #     print(delta_T_val)
#     #     print(poisson_rv)
#     #     print(stab_function)
#     #     print('---')
#     # Limit stability function to 1
#     # if stab_function > 1:
#     #     stab_function = 1
#
#     Ri_list.append(Ri_val)
#     poisson_list.append(poisson_rv)
#     T_list.append(delta_T_val)
#     stab_func_list.append(stab_function)
#     return stab_function
#
#
# def define_ODE_with_stoch_stab_func_poisson(t, delta_T, param):
#     f_stab = define_poisson_stab_function(param, delta_T, param.U)
#     c_D = (param.kappa / np.math.log(param.zr / param.z0)) ** 2
#     return (1 / param.cv) * (
#             param.Q_i - param.Lambda * delta_T - param.rho * param.cp * c_D * param.U * delta_T * f_stab)


# def solve_ODE_with_stoch_stab_func_poisson(param):
#     global Ri_list
#     global poisson_list
#     global T_list
#     global stab_func_list
#     Ri_list = []
#     poisson_list = []
#     T_list = []
#     stab_func_list = []
#
#     solution = solve_ivp(define_ODE_with_stoch_stab_func_poisson, t_span=[param.t_start, param.t_end],
#                          y0=[param.delta_T_0], t_eval=param.t_span, args=(param,))
#
#     print(len(T_list))
#     print(np.shape(solution.y))
#     print(len(param.t_span))
#     fig, ax = plt.subplots(4, 1, figsize=(15, 20))
#     axs = ax.ravel()
#     axs[0].plot(T_list[200:400], "b-", label="solution")
#     axs[1].plot(stab_func_list[200:400], "r-", label="stab_func")
#     axs[2].plot(poisson_list[200:400], "g-", label="poisson")
#     axs[3].scatter(Ri_list[200:400], stab_func_list[200:400])
#
#     axs[0].legend()
#     axs[1].legend()
#     axs[2].legend()
#
#     fig.savefig(f'solution.png', bbox_inches='tight', dpi=300)
#
#     return solution.y.flatten().reshape(-1, 1)


def calculate_d_delta_T_d_t_with_poisson_stab_func(t, delta_T_val, param):
    """Define energy balance model with randomized stability function. It is randomized by adding poisson distributed
    random variables if the Richardson number is higher than the critical number. Otherwise, the stability function
    specified in the parameter file is used.

    Args:
        t (float): time at which the ODE is evaluated
        delta_T_val (float): value of delta T at time step t
        param (class): parameter class which is defined in parameters.py

    Returns:
        (float): value of ddelta_T/dt for time step t
    """
    # Set random seed to ensure that the stochastic process is different for every run (even when they are parallel)
    np.random.seed()
    # Define Poisson random variable
    poisson_rv = np.random.poisson(lam=param.poisson_lambda, size=1) * 0.1
    # Calculate Richardson number
    Ri_val = solve_ODE.calculate_richardson_number(param, delta_T_val, param.U)
    # Calculate stochastic stability function for given Richardson number
    # If the Richardson number is smaller than the critical one use one of the stability functions given by
    # van de Wiel et. al.
    if Ri_val <= param.Ri_c:
        stab_function = solve_ODE.calculate_stability_function(param, delta_T_val, param.U)
    # else use the stochasticly modified one
    else:
        stab_function = solve_ODE.calculate_stability_function(param, delta_T_val, param.U) + poisson_rv

    Ri_list.append(Ri_val)
    poisson_list.append(poisson_rv)
    stab_func_list.append(stab_function)

    c_D = (param.kappa / np.math.log(param.zr / param.z0)) ** 2
    return (1 / param.cv) * (
            param.Q_i - param.Lambda * delta_T_val - param.rho * param.cp * c_D * param.U * delta_T_val * stab_function)


def define_runge_kutta_solver(t0, delta_T_0, params):
    """Define Runge Kutta 4 solver to solve energy balance model.

    Args:
        t0 (float): first time value at which ODE is solved
        delta_T_0 (float): initial value for delta T
        params (class): parameter class which is defined in parameters.py

    Returns:
        numpy array: solution of ODE, i.e. of the energy balance model
    """
    # Define step size for time steps at which the ODE is solved
    h = params.dt
    # Create empty array to store solution of ODE
    delta_T = np.zeros((int(params.t_end),1))
    # Store initial value of delta T
    delta_T[0,:] = delta_T_0
    # Set first time value to t0
    t = t0
    # Solve ODE for all time steps > t0 with Runge Kutta 4 algorithm
    for i in range(1, int(params.t_end)):
        k1 = h * calculate_d_delta_T_d_t_with_poisson_stab_func(t, delta_T[i - 1,:], params)
        k2 = h * calculate_d_delta_T_d_t_with_poisson_stab_func(t + 0.5 * h, delta_T[i - 1,:] + 0.5 * k1, params)
        k3 = h * calculate_d_delta_T_d_t_with_poisson_stab_func(t + 0.5 * h, delta_T[i - 1,:] + 0.5 * k2, params)
        k4 = h * calculate_d_delta_T_d_t_with_poisson_stab_func(t + h, delta_T[i - 1,:] + k3, params)

        # Update next value of y
        delta_T[i,:] = delta_T[i - 1,:] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Update time value
        t = t + h
    return delta_T


def solve_ODE_with_stoch_stab_func_poisson(param):
    """Wrapper function to solve energy balance model with random stability function (randomized with a poisson
    distributed random variable) with a Runge Kutta algorithm of order 4.

    Args:
        param (class): parameter class which is defined in parameters.py

    Returns:
        numpy array: solution of energy balance model with random stability function
    """
    global Ri_list
    global poisson_list
    global stab_func_list
    Ri_list = []
    poisson_list = []
    stab_func_list = []

    solution = define_runge_kutta_solver(param.t_start, param.delta_T_0, param)

    fig, ax = plt.subplots(4, 1, figsize=(15, 20))
    axs = ax.ravel()
    axs[0].plot(solution, "b-", label="solution")
    axs[1].plot(stab_func_list, "r-", label="stab_func")
    axs[2].plot(poisson_list, "g-", label="poisson")
    axs[3].scatter(Ri_list, stab_func_list)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    fig.savefig(f'solution.png', bbox_inches='tight', dpi=300)

    return solution
