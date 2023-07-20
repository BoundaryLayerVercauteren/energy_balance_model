"""Script for all functions related to the deterministic model (eq. 2)."""
import numpy as np
import sdeint
from scipy.integrate import quad, solve_ivp


def calculate_neutral_drag_coefficient(param):
    return (param.kappa / np.math.log(param.zr / param.z0)) ** 2


def calculate_richardson_number(param, delta_T, U):
    return param.zr * (param.grav / param.Tr) * (delta_T / (U**2))


def calculate_stability_function(param, delta_T, U):
    """Calculate stability function (short- or long-tail). The type is specified in the parameter class."""
    # Calculate Richardson number
    Ri = calculate_richardson_number(param, delta_T, U)
    # Calculate stability function
    if param.stab_func_type == "short_tail":
        return np.math.exp(-2 * param.alpha * Ri - (param.alpha * Ri) ** 2)
    elif param.stab_func_type == "long_tail":
        return np.math.exp(-2 * param.alpha * Ri)
    else:
        print("The stability function needs to be either short_tail or long_tail.")
        exit()


def define_ODE(t, delta_T, u, Lambda, Qi, param, f_stab=None):
    """Define conceptual model for temperature inversions (eq. 2) (without perturbations)."""
    if f_stab is None:
        f_stab = calculate_stability_function(param, delta_T, u)
    c_D = calculate_neutral_drag_coefficient(param)
    return (1 / param.cv) * (
        Qi - Lambda * delta_T - param.rho * param.cp * c_D * u * delta_T * f_stab
    )


def solve_ODE(param):
    """Solve conceptual model for temperature inversions (eq. 2) (without perturbations)."""
    return solve_ivp(
        define_ODE,
        [param.t_start, param.t_end],
        [param.delta_T_0],
        t_eval=param.t_span,
        args=(param.U, param.Lambda, param.Q_i, param),
    )


def solve_ODE_with_time_dependent_u(param):
    """Define and solve conceptual model for temperature inversions (eq. 2) with time-varying wind forcing u."""

    # A stochastic solver (but without a stochastic term) is used to make results comparable with the ones where the
    # stability function is randomized (see figure 10).
    def _f(delta_T, t):
        # Find index of time step in t_span which is closest to the time step at which the ODE is currently evaluated
        idx = (np.abs(param.t_span - t)).argmin()
        # Find corresponding u value
        param.U = param.u_range[idx]
        return define_ODE(t, delta_T, param.U, param.Lambda, param.Q_i, param)

    def _G(X, t):
        return 0

    return sdeint.itoint(_f, _G, param.delta_T_0, param.t_span)


def calculate_potential(delta_T, u, param):
    """Calculate potential of conceptual model for temperature inversions (eq. 2). See section 2.2. for further
    details."""
    ode = lambda x: define_ODE(0.0, x, u, param.Lambda, param.Q_i, param)
    potential = np.zeros_like(delta_T)
    for i, val in enumerate(delta_T):
        y, _ = quad(ode, 0, val)
        potential[i] = y
    return potential
