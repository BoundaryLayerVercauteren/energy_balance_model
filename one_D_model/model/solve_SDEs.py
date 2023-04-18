import numpy as np
import sdeint

from one_D_model.model import solve_ODE


def define_noise_term(delta_T, t, sigma):
    return sigma


def solve_SDE(param):
    """Original model by van de Wiel with additive noise term"""
    # Note: itoint(f, G, y0, tspan) for Ito equation dy = f(y,t)dt + G(y,t)dW
    f = lambda delta_T, t: solve_ODE.define_deterministic_ODE(t, delta_T, param.U, param.Lambda, param.Q_i, param)
    G = lambda delta_T, t: define_noise_term(delta_T, t, param.sigma_delta_T)
    return sdeint.itoint(f, G, param.delta_T_0, param.t_span)


def solve_SDE_with_stoch_u(param):
    """Original model by van de Wiel with stochastic wind equation"""
    # Combine initial conditions for
    initial_cond = np.array([param.delta_T_0, param.U])

    # Define functions for 2D SDE
    def _f(X, t):
        return np.array([solve_ODE.define_deterministic_ODE(t, X[0], X[1], param.Lambda, param.Q_i, param),
                         param.relax_u * (X[1] - param.mu_u)])

    def _G(X, t):
        return np.diag([0.0, define_noise_term(X[0], t, param.sigma_u)])

    return sdeint.itoint(_f, _G, initial_cond, param.t_span)


def solve_SDE_with_stoch_Qi(param):
    """Original model by van de Wiel with stochastic cloud cover (hidden in Q_i) equation"""
    # Combine initial conditions for
    initial_cond = np.array([param.delta_T_0, param.Q_i])

    # Define functions for 2D SDE
    def _f(X, t):
        return np.array([solve_ODE.define_deterministic_ODE(t, X[0], param.U, param.Lambda, X[1], param),
                         param.relax * (X[1] - param.Q_i)])

    def _G(X, t):
        return np.diag([0.0, define_noise_term(X[0], t, param.sigma_Q_i)])

    return sdeint.itoint(_f, _G, initial_cond, param.t_span)


def solve_SDE_with_stoch_lambda(param):
    """Original model by van de Wiel with stochastic lambda equation"""
    # Combine initial conditions for
    initial_cond = np.array([param.delta_T_0, param.Lambda])

    # Define functions for 2D SDE
    def _f(X, t):
        return np.array([solve_ODE.define_deterministic_ODE(t, X[0], param.U, X[1], param.Q_i, param),
                         param.relax * (X[1] - param.Lambda)])

    def _G(X, t):
        return np.diag([0.0, define_noise_term(X[0], t, param.sigma_lambda)])

    return sdeint.itoint(_f, _G, initial_cond, param.t_span)


def solve_SDE_with_stoch_u_and_internal_var(param):
    """Energy balance model with Ornstein Uhlenbeck process to account for random wind and additive noise
    for internal variability."""
    # Define initial conditions
    initial_cond = np.array([param.delta_T_0, param.U])

    # Define functions for 2D SDE
    def _f(X, t):
        return np.array([solve_ODE.define_deterministic_ODE(t, X[0], X[1], param.Lambda, param.Q_i, param),
                         param.relax_u * (X[1] - param.mu_u)])

    def _G(X, t):
        return np.diag([param.sigma_delta_T, param.sigma_u])

    return sdeint.itoint(_f, _G, initial_cond, param.t_span)
