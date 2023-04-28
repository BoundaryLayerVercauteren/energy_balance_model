import numpy as np
import sdeint

from one_D_model.model import solve_ODE
from one_D_model.model.solve_ODE import calculate_neutral_drag_coefficient, calculate_stability_function


def define_SDE(t, delta_T, f_stab, param):
    c_D = calculate_neutral_drag_coefficient(param)
    return (1 / param.cv) * (
            param.Q_i - param.Lambda * delta_T - param.rho * param.cp * c_D * param.U * delta_T * f_stab)


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
