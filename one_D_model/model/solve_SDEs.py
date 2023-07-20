"""Script where all stochastic differential equations (SDEs) are defined and solved."""

import numpy as np
import sdeint

from one_D_model.model import solve_ODE
from one_D_model.model.solve_ODE import calculate_stability_function


def solve_SDE_with_internal_var(param):
    """Conceptual model for temperature inversions with additive noise term to represent small-scale fluctuations of an
    unresolved process (eq. 3)."""
    # Note: itoint(f, G, y0, tspan) for Ito equation dy = f(y,t)dt + G(y,t)dW
    f = lambda delta_T, t: solve_ODE.define_ODE(
        t, delta_T, param.U, param.Lambda, param.Q_i, param
    )
    G = lambda delta_T, t: param.sigma_delta_T
    return sdeint.itoint(f, G, param.delta_T_0, param.t_span)


def solve_SDE_with_stoch_u(param):
    """Conceptual model for temperature inversions with an Ornstein-Uhlenbeck process representing small-scale
    fluctuations of the wind forcing (eq. 4)."""
    # Combine initial conditions for the temperature difference and wind velocity
    initial_cond = np.array([param.delta_T_0, param.U])

    # Define deterministic part
    def _f(X, t):
        return np.array(
            [
                solve_ODE.define_ODE(t, X[0], X[1], param.Lambda, param.Q_i, param),
                param.relax_u * (X[1] - param.U),
            ]
        )

    # Define stochastic part
    def _G(X, t):
        return np.diag([0.0, param.sigma_u])

    return sdeint.itoint(_f, _G, initial_cond, param.t_span)


def solve_SDE_with_stoch_u_and_internal_var(param):
    """Conceptual model for temperature inversions with an Ornstein-Uhlenbeck process representing small-scale
    fluctuations of the wind forcing and additive noise representing  small-scale fluctuations of an
    unresolved process (eq. 5)."""
    # Combine initial conditions for the temperature difference and wind velocity
    initial_cond = np.array([param.delta_T_0, param.U])

    # Define deterministic part
    def _f(X, t):
        return np.array(
            [
                solve_ODE.define_ODE(t, X[0], X[1], param.Lambda, param.Q_i, param),
                param.relax_u * (X[1] - param.U),
            ]
        )

    # Define stochastic part
    def _G(X, t):
        return np.diag([param.sigma_delta_T, param.sigma_u])

    return sdeint.itoint(_f, _G, initial_cond, param.t_span)


def solve_SDE_with_stoch_stab_function(param):
    """Conceptual model for temperature inversions with multiplicative noise representing turbulent bursts (eq. 6)."""
    # Combine initial conditions for the temperature difference and stability function (given by either the long- or
    # short-tail stability function (section 2.2.)
    phi_0 = calculate_stability_function(param, param.delta_T_0, param.U)
    initial_cond = np.array([param.delta_T_0, phi_0])

    # Define deterministic part
    def _f(X, t):
        return np.array(
            [
                solve_ODE.define_ODE(
                    t, X[0], param.U, param.Lambda, param.Q_i, param, X[1]
                ),
                param.relax_phi
                * (X[1] - calculate_stability_function(param, X[0], param.U)),
            ]
        )

    # Define stochastic part
    def _G(X, t):
        # Calculate Richardson number
        Ri = solve_ODE.calculate_richardson_number(param, X[0], param.U)
        # Choose the noise strength based on the current Richardson number
        if Ri > param.Ri_c:
            sigma = param.sigma_phi
        else:
            sigma = 0
        return np.diag([0.0, sigma * X[1]])

    return sdeint.itoint(_f, _G, initial_cond, param.t_span)


def solve_SDE_with_stoch_stab_function_time_dependent_u(param):
    """Conceptual model for temperature inversions with multiplicative noise representing turbulent bursts (eq. 6) and
    time-varying wind forcing U."""
    #  Combine initial conditions for the temperature difference and stability function (given by either the long- or
    # short-tail stability function (section 2.2.)
    phi_0 = calculate_stability_function(param, param.delta_T_0, param.U)
    initial_cond = np.array([param.delta_T_0, phi_0])

    # Define deterministic part
    def _f(X, t):
        # Find index of time step in t_span which is closest to the time step at which the SDE is currently evaluated
        idx = (np.abs(param.t_span - t)).argmin()
        # Find corresponding u value
        param.U = param.u_range[idx]
        return np.array(
            [
                solve_ODE.define_ODE(
                    t, X[0], param.U, param.Lambda, param.Q_i, param, X[1]
                ),
                param.relax_phi
                * (X[1] - calculate_stability_function(param, X[0], param.U)),
            ]
        )

    # Define stochastic part
    def _G(X, t):
        # Find index of time step in t_span which is closest to the time step at which the SDE is currently evaluated
        idx = (np.abs(param.t_span - t)).argmin()
        # Find corresponding u value
        param.U = param.u_range[idx]
        # Calculate Richardson number
        Ri = solve_ODE.calculate_richardson_number(param, X[0], param.U)
        # Choose the noise strength based on the current Richardson number
        if Ri > param.Ri_c:
            sigma = param.sigma_phi
        else:
            sigma = 0
        return np.diag([0.0, sigma * X[1]])

    return sdeint.itoint(_f, _G, initial_cond, param.t_span)
