from dataclasses import dataclass
from dataclasses_json import dataclass_json
import math

from one_D_model.model import solve_ODE


@dataclass_json
@dataclass
class Parameters:
    kappa: float = 0.4
    zr: float = 10
    z0: float = 0.01
    grav: float = 10
    Tr: float = 2


def test_calculate_neutral_drag_coefficient():
    # Set up parameter class
    param = Parameters()
    assert math.isclose(solve_ODE.calculate_neutral_drag_coefficient(param), 0.0033530968357620263)


def test_calculate_richardson_number():
    # Set up parameter class
    param = Parameters()
    assert math.isclose(solve_ODE.calculate_richardson_number(param, 22.5, 3), 125.0)
