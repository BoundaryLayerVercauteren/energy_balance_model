"""File where values for all parameters which are used by the model can be set."""

from dataclasses import dataclass

import numpy as np
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Parameters:
    Q_i: float = 50.0  # isothermal net radiation [W/m^2]
    Lambda: float = 2.0  # lumped parameter representing all feedback from soil heat conduction and radiative cooling as a net linear effect [W/m^2K]
    kappa: float = 0.4  # von Karman constant [-]
    cv: float = 1000  # heat capacity of the soil per surface area [J/m^2K]
    zr: float = 10.0  # reference height [m]
    Tr: float = 243.0  # temperature at reference height [K]
    rho: float = 1.0  # density of air at constant pressure [kg/m^3]
    cp: float = 1005.0  # heat capacity of air at constant pressure [J/kg^3K]
    grav: float = 9.81  # gravitational constant
    z0: float = 0.01  # roughness length [m]
    U: float = 5.6  # wind speed at reference height [m/s]
    alpha: float = 5.0  # parameter for stability function

    stab_func_type: str = 'short_tail'  # stability function type, options are [short_tail, long_tail]

    t_start: float = 0.0  # simulation start time
    t_end_h: float = 1 * 1.0  # simulation end time [hours]
    t_end: float = t_end_h * 3600  # simulation end time [seconds]
    dt: float = 1  # size of time steps [seconds]
    num_steps: float = int(t_end / dt)  # number of steps in time
    t_span: np.ndarray = np.linspace(t_start, t_end, num_steps)
    t_span_h: np.ndarray = np.linspace(t_start, t_end_h, num_steps)

    delta_T_0: float = 24  # initial condition for delta T

    Ri_c: float = 0.25  # critical Richardson number

    sigma_delta_T: float = 0.1  # sigma for delta T noise term (eq. 3 and 5)
    sigma_u: float = 0.005  # sigma for u noise term (eq. 4 and 5)
    sigma_phi: float = 0.1  # sigma for stability function (eq. 6)

    relax_u: float = -0.005  # coefficient of relaxation to equilibrium (eq. 4 and 5)
    relax_phi: float = -0.005  # coefficient of relaxation to equilibrium (eq. 6)

    # values of u for SDE if u is time dependent
    u_range_start: float = 4.5
    u_range_end: float = 4.6

    num_simulation: int = 2  # number of runs for Monte Carlo simulation
    num_proc: int = 2  # number of processes to be used in parallelization
