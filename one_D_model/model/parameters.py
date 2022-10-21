import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Parameters:
    Q_i: float = 50.0  # Site dependant
    Lambda: float = 2.0  # Site dependant
    kappa: float = 0.4  # von Karman
    cv: float = 1000  # very uncertain: heat capacity of the soil per surface area - no value in the paper
    zr: float = 10.0
    Tr: float = 243.0
    rho: float = 1.0
    cp: float = 1005.0
    alpha: float = 5.0
    grav: float = 9.81
    z0: float = 0.01  # roughness - varies with surface roughness type, uncertain but not so variable in time
    U: float = 5.6  # Geostrophic wind - main bifurcation parameter

    stab_func_type: str = 'short_tail'

    t_start: float = 0.0
    t_end_h: float = 24.0  # in hours
    t_end: float = t_end_h * 3600  # in seconds
    dt: float = 1  # seconds, size of time steps
    num_steps: float = int(t_end / dt)  # number of steps in time
    t_span: np.ndarray = np.linspace(t_start, t_end, num_steps)
    t_span_h: np.ndarray = np.linspace(t_start, t_end_h, num_steps)

    delta_T_0: float = 25  # initial condition for delta T

    sigma_delta_T: float = 0.3  # sigma for delta T noise term
    sigma_u: float = 0.2  # sigma for u noise term
    sigma_Q_i: float = 0.7  # sigma for Q_i noise term
    sigma_lambda: float = 0.1  # sigma for lambda noise term
    sigma_z0: float = 0.05  # sigma for z0 noise term
    mu_z0: float = 0.001

    relax: float = 0.005  # coefficient of relaxation to equilibrium

    num_simulation: int = 1000  # number of runs for Monte Carlo simulation
    num_proc: int = 5
