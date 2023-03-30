import numpy as np
import sdeint

from one_D_model.model import solve_ODE
from one_D_model.model.solve_ODE import calculate_neutral_drag_coefficient, calculate_stability_function


# First define the scaling functions that scale the SDE parameter values with Ri
def kappa(x):
    # Coefficients of the Kappa function
    k1 = 9.3212
    k2 = 0.9088
    k3 = 0.0738
    k4 = 8.3220
    return k1 * np.tanh(k2 * np.log10(x) - k3) + k4


def upsilon(x):
    # Coefficients of the Upsilon function
    u1 = 0.4294
    u2 = 0.1749
    return 10 ** (u1 * np.log10(x) + u2)


def sigma(x, sigma_s):
    # Coefficients of the Sigma function
    s1 = 0.8069
    s2 = 0.6044
    s3 = 0.8368
    return 10 ** (s1 * np.tanh(s2 * np.log10(x) - s3) + sigma_s)


def define_SDE(t, delta_T, phi_stoch, param):
    f_stab = 1 / phi_stoch
    c_D = calculate_neutral_drag_coefficient(param)
    return (1 / param.cv) * (
            param.Q_i - param.Lambda * delta_T - param.rho * param.cp * c_D * param.U * delta_T * f_stab)


def define_stoch_stab_function(delta_T, phi_stoch, t, param):
    Ri = solve_ODE.calculate_richardson_number(param, delta_T, param.U)
    return (1 / 3600) * (1 + kappa(Ri) * phi_stoch - upsilon(Ri) * phi_stoch ** 2)


def solve_SDE_with_stoch_stab_function(param):
    """Original model by van de Wiel with stochastic stability function given by Boyko et al., 2023."""
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
    # van de Wiel et al.
    if Ri_val <= param.Ri_c:
        stab_function = solve_ODE.calculate_stability_function(param, delta_T_val, param.U)
    # else use the stochasticly modified one
    else:
        stab_function = solve_ODE.calculate_stability_function(param, delta_T_val, param.U) + poisson_rv

    c_D = calculate_neutral_drag_coefficient(param)

    return (1 / param.cv) * (
            param.Q_i - param.Lambda * delta_T_val - param.rho * param.cp * c_D * param.U * delta_T_val * stab_function)


def define_runge_kutta_solver(t0, delta_T_0, params, d_delta_T_d_t=calculate_d_delta_T_d_t_with_poisson_stab_func):
    """Define Runge Kutta 4 solver to solve energy balance model.

    Args:
        t0 (float): first time value at which ODE is solved
        delta_T_0 (float): initial value for delta T
        params (class): parameter class which is defined in parameters.py
        d_delta_T_d_t (function): name of function which defines the ODE that will be solved

    Returns:
        numpy array: solution of ODE, i.e. of the energy balance model
    """
    # Define step size for time steps at which the ODE is solved
    h = params.dt
    # Create empty array to store solution of ODE
    delta_T = np.zeros(params.num_steps)
    # Store initial value of delta T
    delta_T[0] = delta_T_0
    # Set first time value to t0
    t = t0
    # Solve ODE for all time steps > t0 with Runge Kutta 4 algorithm
    for idx in range(0, params.num_steps - 1):
        k1 = d_delta_T_d_t(t, delta_T[idx], params)
        k2 = d_delta_T_d_t(t + 0.5 * h, delta_T[idx] + 0.5 * k1 * h, params)
        k3 = d_delta_T_d_t(t + 0.5 * h, delta_T[idx] + 0.5 * k2 * h, params)
        k4 = d_delta_T_d_t(t + h, delta_T[idx] + k3 * h, params)

        # Update next value of y
        delta_T[idx + 1] = delta_T[idx] + h * (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Update time value
        t = t + h

    return delta_T.flatten()


def solve_ODE_with_stoch_stab_func_poisson(param):
    """Wrapper function to solve energy balance model with random stability function (randomized with a poisson
    distributed random variable) with a Runge Kutta algorithm of order 4.

    Args:
        param (class): parameter class which is defined in parameters.py

    Returns:
        numpy array: solution of energy balance model with random stability function
    """
    return define_runge_kutta_solver(param.t_start, param.delta_T_0, param)


def solve_SDE_with_stoch_stab_function_multi_noise(param):
    """Original model by van de Wiel with multiplicative noise in stochastic stability function."""
    # Combine initial conditions for
    phi_0 = 1.0
    initial_cond = np.array([param.delta_T_0, phi_0])

    # Define functions for 2D SDE
    def _f(X, t):
        return np.array([define_SDE(t, X[0], X[1], param),
                         param.relax_phi * (X[1] - calculate_stability_function(param, X[0], param.U))])

    def _G(X, t):
        Ri = solve_ODE.calculate_richardson_number(param, X[0], param.U)
        if Ri > param.Ri_c:
            sigma = param.sigma_phi
        else:
            sigma = 0
        return np.diag([0.0, sigma * X[1]])

    return sdeint.itoint(_f, _G, initial_cond, param.t_span)
