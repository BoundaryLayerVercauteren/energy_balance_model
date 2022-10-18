import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve


def calculate_richardson_number(param, delta_T, U):
    return param.zr * param.grav / param.Tr * delta_T / U ** 2


def calculate_stability_function(param, delta_T, U):
    # Calculate Richardson number
    Ri = calculate_richardson_number(param, delta_T, U)
    # Calculate stability function
    if param.stab_func_type == 'short_tail':
        return np.math.exp(-2 * param.alpha * Ri - (param.alpha * Ri) ** 2)
    elif param.stab_func_type == 'long_tail':
        return np.math.exp(-2 * param.alpha * Ri)


def define_deterministic_ODE(t, delta_T, u, Lambda, Qi, z0, param):
    f_stab = calculate_stability_function(param, delta_T, u)
    c_D = (param.kappa / np.math.log(param.zr / z0)) ** 2
    return (1/param.cv) * (Qi - Lambda * delta_T - param.rho * param.cp * c_D * u * delta_T * f_stab)


def solve_deterministic_ODE(param):
    return solve_ivp(define_deterministic_ODE, [param.t_start, param.t_end], [param.delta_T_0], t_eval=param.t_span,
                     args=(param.U, param.Lambda, param.Q_i, param.z0, param))


def calculate_potential(delta_T, u, param):

    ode = lambda x: define_deterministic_ODE(0.0, x, u, param.Lambda, param.Q_i, param.z0, param)

    potential = np.zeros_like(delta_T)
    for i, val in enumerate(delta_T):
        y, _ = quad(ode, 0, val)
        potential[i] = y
    return potential





