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


def define_SDE(t, delta_T, f_stab, param):
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
        return np.array([define_SDE(t, X[0], 1/X[1], param), define_stoch_stab_function(X[0], X[1], t, param)])

    def _G(X, t):
        Ri = solve_ODE.calculate_richardson_number(param, param.delta_T_0, param.U)
        return np.diag([0.0, (1 / np.sqrt(3600)) * (sigma(Ri, param.sigma_s) * X[1])])

    result = sdeint.itoint(_f, _G, initial_cond, param.t_span)

    # phi = 1/ regular stability function
    result[:, 1] = 1 / result[:, 1]

    return result


def solve_SDE_with_stoch_stab_function_multi_noise(param):
    """Energy balance model with multiplicative noise in stochastic stability function."""
    # Combine initial conditions for
    phi_0 = calculate_stability_function(param, param.delta_T_0, param.U)
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


def solve_SDE_with_stoch_stab_function_multi_noise_time_dependent_u(param):
    """Energy balance model with multiplicative noise in stochastic stability function."""
    # Combine initial conditions for
    phi_0 = calculate_stability_function(param, param.delta_T_0, param.U)
    initial_cond = np.array([param.delta_T_0, phi_0])

    # Define functions for 2D SDE
    def _f(X, t):
        # Find index of time step in t_span which is closest to the time step at which the SDE is currently evaluated
        idx = (np.abs(param.t_span - t)).argmin()
        # Find corresponding u value
        param.U = param.u_range[idx]
        return np.array([define_SDE(t, X[0], X[1], param),
                         param.relax_phi * (X[1] - calculate_stability_function(param, X[0], param.U))])

    def _G(X, t):
        # Find index of time step in t_span which is closest to the time step at which the SDE is currently evaluated
        idx = (np.abs(param.t_span - t)).argmin()
        # Find corresponding u value
        param.U = param.u_range[idx]
        # Calculate Richardson number
        Ri = solve_ODE.calculate_richardson_number(param, X[0], param.U)
        if Ri > param.Ri_c:
            sigma = param.sigma_phi
        else:
            sigma = 0
        return np.diag([0.0, sigma * X[1]])

    return sdeint.itoint(_f, _G, initial_cond, param.t_span)