import sdeint
import numpy as np
import scipy

from one_D_model.model import solve_ODE


def define_noise_term(delta_T, t, sigma):
    return sigma


def solve_SDE(param):
    """Original model by van de Wiel with additive noise term"""
    # Note: itoint(f, G, y0, tspan) for Ito equation dy = f(y,t)dt + G(y,t)dW
    f = lambda delta_T, t: solve_ODE.define_deterministic_ODE(t, delta_T, param.U, param.Lambda, param.Q_i, param.z0, param)
    G = lambda delta_T, t: define_noise_term(delta_T, t, param.sigma_delta_T)
    return sdeint.itoint(f, G, param.delta_T_0, param.t_span)


def solve_SDE_with_stoch_u(param):
    """Original model by van de Wiel with stochastic wind equation"""
    # Combine initial conditions for
    initial_cond = np.array([param.delta_T_0, param.U])

    # Define functions for 2D SDE
    def _f(X, t):
        return np.array([solve_ODE.define_deterministic_ODE(t, X[0], X[1], param.Lambda, param.Q_i, param.z0, param), -param.relax * (X[1] - param.U)])

    def _G(X, t):
        return np.diag([0.0, define_noise_term(X[0], t, param.sigma_u)])

    return sdeint.itoint(_f, _G, initial_cond, param.t_span)


def solve_SDE_with_stoch_Qi(param):
    """Original model by van de Wiel with stochastic cloud cover (hidden in Q_i) equation"""
    # Combine initial conditions for
    initial_cond = np.array([param.delta_T_0, param.Q_i])

    # Define functions for 2D SDE
    def _f(X, t):
        return np.array([solve_ODE.define_deterministic_ODE(t, X[0], param.U, param.Lambda, X[1], param.z0, param), -param.relax * (X[1] - param.Q_i)])

    def _G(X, t):
        return np.diag([0.0, define_noise_term(X[0], t, param.sigma_Q_i)])

    return sdeint.itoint(_f, _G, initial_cond, param.t_span)


def solve_SDE_with_stoch_lambda(param):
    """Original model by van de Wiel with stochastic lambda equation"""
    # Combine initial conditions for
    initial_cond = np.array([param.delta_T_0, param.Lambda])

    # Define functions for 2D SDE
    def _f(X, t):
        return np.array([solve_ODE.define_deterministic_ODE(t, X[0], param.U, X[1], param.Q_i, param.z0, param), -param.relax * (X[1] - param.Lambda)])

    def _G(X, t):
        return np.diag([0.0, define_noise_term(X[0], t, param.sigma_lambda)])

    return sdeint.itoint(_f, _G, initial_cond, param.t_span)


def solve_SDE_with_stoch_z0(_, param):
    """Original model by van de Wiel with stochastic roughness length equation"""

    z0_list = []
    def _z0(t):
        z_0 = scipy.stats.lognorm.rvs(s=param.sigma_z0, loc=param.mu_z0, scale=param.sigma_z0)
        z0_list.append([t,z_0])
        return z_0

    # Define SDE
    def _define_SDE(t, delta_T):
        f_stab = solve_ODE.calculate_stability_function(param, delta_T, param.U)
        c_D = (param.kappa / np.math.log(param.zr / _z0(t))) ** 2
        return (1 / param.cv) * (param.Q_i - param.Lambda * delta_T - param.rho * param.cp * c_D * param.U * delta_T * f_stab)

    try:
        result = scipy.integrate.solve_ivp(_define_SDE, [param.t_start, param.t_end], [param.delta_T_0], t_eval=param.t_span)
        return result.t.flatten(), result.y.flatten(), z0_list
    except Exception:
        return np.nan, np.nan, np.nan

