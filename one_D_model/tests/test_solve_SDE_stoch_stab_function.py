from dataclasses import dataclass
from dataclasses_json import dataclass_json
import numpy as np

from one_D_model.model import solve_SDE_stoch_stab_function


@dataclass_json
@dataclass
class Parameters:
    dt: float = 1
    num_steps: int = 10


def dxdt(t, x, param):
    return 2*t


def test_define_runge_kutta_solver():
    # Set up parameter class
    param = Parameters()

    solution_rk = solve_SDE_stoch_stab_function.define_runge_kutta_solver(0, 0, param, dxdt)
    solution_analytic = np.arange(0, param.num_steps, param.dt) ** 2

    assert np.allclose(solution_rk, solution_analytic, rtol=1e-05, atol=1e-08)
