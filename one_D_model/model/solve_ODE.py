import numpy as np
from scipy.integrate import quad, solve_ivp


def calculate_neutral_drag_coefficient(param):
    """Function to calculate neutral drag coefficient for given reference height, surface roughness length, von karman
    constant

    Args:
        param (class): parameter class which is defined in parameters.py

    Returns:
        (float): neutral drag coefficient
    """
    return (param.kappa / np.math.log(param.zr / param.z0)) ** 2


def calculate_richardson_number(param, delta_T, U):
    """Function to calculate Richardson number

    Args:
        param (class): parameter class which is defined in parameters.py
        delta_T (float): potential temperature
        U (float): wind velocity

    Returns:
        (float): Richardson number
    """
    return param.zr * (param.grav / param.Tr) * (delta_T / (U ** 2))


def calculate_stability_function(param, delta_T, U):
    """Function to calculate stability function (short- or long-tail). The type is specified in the parameter class.

    Args:
        param (class): parameter class which is defined in parameters.py
        delta_T (float): potential temperature
        U (float): wind velocity

    Returns:
        (float): values of stability function
    """
    # Calculate Richardson number
    Ri = calculate_richardson_number(param, delta_T, U)
    # Calculate stability function
    if param.stab_func_type == 'short_tail':
        return np.math.exp(-2 * param.alpha * Ri - (param.alpha * Ri) ** 2)
    elif param.stab_func_type == 'long_tail':
        return np.math.exp(-2 * param.alpha * Ri)
    else:
        print('The stability function needs to be either sort_tail or long_tail.')
        exit()


def define_deterministic_ODE(t, delta_T, u, Lambda, Qi, param):
    f_stab = calculate_stability_function(param, delta_T, u)
    c_D = calculate_neutral_drag_coefficient(param)
    return (1/param.cv) * (Qi - Lambda * delta_T - param.rho * param.cp * c_D * u * delta_T * f_stab)


def solve_deterministic_ODE(param):
    return solve_ivp(define_deterministic_ODE, [param.t_start, param.t_end], [param.delta_T_0], t_eval=param.t_span,
                     args=(param.U, param.Lambda, param.Q_i, param))

def define_deterministic_ODE_var_u(t, delta_T, u_range, Lambda, Qi, param):
    # Find index of time step in t_span which is closest to the time step at which the SDE is currently evaluated
    idx = (np.abs(param.t_span - t)).argmin()
    # Find corresponding u value
    u = u_range[idx]
    print(u)
    f_stab = calculate_stability_function(param, delta_T, u)
    c_D = calculate_neutral_drag_coefficient(param)
    return (1/param.cv) * (Qi - Lambda * delta_T - param.rho * param.cp * c_D * u * delta_T * f_stab)


def solve_deterministic_ODE_var_u(param):
    return solve_ivp(define_deterministic_ODE_var_u, [param.t_start, param.t_end_h], [param.delta_T_0],
                     t_eval=param.t_span, args=(param.u_range, param.Lambda, param.Q_i, param))


def calculate_potential(delta_T, u, param):

    ode = lambda x: define_deterministic_ODE(0.0, x, u, param.Lambda, param.Q_i, param)

    potential = np.zeros_like(delta_T)
    for i, val in enumerate(delta_T):
        y, _ = quad(ode, 0, val)
        potential[i] = y
    return potential





