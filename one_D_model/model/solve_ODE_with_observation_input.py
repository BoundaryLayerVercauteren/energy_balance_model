import numpy as np
from scipy.integrate import solve_ivp

from one_D_model.model.solve_ODE import calculate_stability_function


def get_u_at_time_t(t, observed_u, t_span):
    t_idx = (np.abs(t_span - t)).argmin()
    return observed_u[t_idx]


def define_deterministic_ODE(t, delta_T, Lambda, Qi, z0, param, data_u):
    u = get_u_at_time_t(t, data_u, param.t_span)
    f_stab = calculate_stability_function(param, delta_T, u)
    c_D = (param.kappa / np.math.log(param.zr / z0)) ** 2

    return (1 / param.cv) * (Qi - Lambda * delta_T - param.rho * param.cp * c_D * u * delta_T * f_stab)


def solve_deterministic_ODE(param, input_u):
    solution = solve_ivp(define_deterministic_ODE, t_span=[param.t_start, param.t_end], y0=[param.delta_T_0],
                         t_eval=param.t_span,
                         args=(param.Lambda, param.Q_i, param.z0, param, input_u))
    return np.array([solution.t.flatten(), solution.y.flatten()])
