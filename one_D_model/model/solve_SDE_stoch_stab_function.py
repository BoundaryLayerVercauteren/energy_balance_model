import numpy as np
import sdeint
from one_D_model.model import solve_ODE


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
    """Original model by van de Wiel with stochastic stability function"""
    # Combine initial conditions for
    phi_0 = 1 / solve_ODE.calculate_stability_function(param, param.delta_T_0, param.U)
    initial_cond = np.array([param.delta_T_0, phi_0])

    # Define functions for 2D SDE
    def _f(X, t):
        return np.array([define_SDE(t, X[0], X[1], param), define_stoch_stab_function(X[0], X[1], t, param)])

    def _G(X, t):
        Ri = solve_ODE.calculate_richardson_number(param, param.delta_T_0, param.U)
        return np.diag([0.0, (1 / np.sqrt(3600)) * (sigma(Ri, param.sigma_s) * X[1])])

    return sdeint.itoint(_f, _G, initial_cond, param.t_span)
