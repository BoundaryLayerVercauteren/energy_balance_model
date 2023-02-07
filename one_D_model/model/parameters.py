import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Parameters:
    Q_i: float = 50.0  # isothermal net radiation [W/m^2]
    Lambda: float = 2.0  # lumped parameter representing all feedback from soil heat conduction and radiative cooling as a net linear effect [W/m^2K]
    kappa: float = 0.4  # von Karman constant [-]
    cv: float = 1000  # heat capacity of the soil per surface area [?]
    zr: float = 10.0  # reference height [m]
    Tr: float = 243.0  # temperature at reference height [K]
    rho: float = 1.0  # density of air at constant pressure [kg/m^3]
    cp: float = 1005.0  # heat capacity of air at constant pressure [J/kg^3K]
    grav: float = 9.81  # gravitational constant
    z0: float = 0.01  # roughness length [m]
    U: float = 5.6  # wind speed at reference height [m/s]
    alpha: float = 5.0  # parameter for stability function

    stab_func_type: str = 'short_tail'  # stability function type [short_tail, long_tail]

    t_start: float = 0.0  # simulation start time
    t_end_h: float = 7*24.0  # simulation end time [hours]
    t_end: float = t_end_h * 3600  # simulation end time [seconds]
    dt: float = 1800  # size of time steps [seconds]
    num_steps: float = int(t_end / dt)  # number of steps in time
    t_span: np.ndarray = np.linspace(t_start, t_end, num_steps)
    t_span_h: np.ndarray = np.linspace(t_start, t_end_h, num_steps)

    delta_T_0: float = 24  # initial condition for delta T

    sigma_delta_T: float = 0.3  # sigma for delta T noise term
    sigma_u: float = 0.66  # sigma for u noise term
    sigma_Q_i: float = 0.7  # sigma for Q_i noise term
    sigma_lambda: float = 0.05  # sigma for lambda noise term
    sigma_z0: float = 0.1  # sigma for z0 noise term
    mu_z0: float = 0.001  # mu for z0 noise term
    sigma_s: float = -0.1

    relax: float = -0.005  # coefficient of relaxation to equilibrium
    relax_u: float = -0.12
    mu_u: float = 5.61

    num_simulation: int = 100  # number of runs for Monte Carlo simulation
    num_proc: int = 4  # number of processes to be used in parallelization
