import math

import numpy as np
import pytest

from one_D_model.model.solve_ODE import (calculate_neutral_drag_coefficient,
                                         calculate_richardson_number,
                                         calculate_stability_function,
                                         solve_ODE)


# Define a mock parameter class for testing purposes
class MockParameters:
    kappa = 0.4
    zr = 100.0
    z0 = 0.01
    grav = 9.81
    Tr = 300.0
    stab_func_type = "short_tail"
    alpha = 0.1
    cv = 1000.0
    rho = 1.2
    cp = 1005.0
    U = 10.0
    Lambda = 0.01
    Q_i = 100.0
    t_start = 0.0
    t_end = 10.0
    t_span = np.linspace(t_start, t_end, 100)
    delta_T_0 = 2.0
    u_range = np.linspace(5.0, 15.0, 100)


# Test cases for the 'calculate_neutral_drag_coefficient' function
def test_calculate_neutral_drag_coefficient():
    param = MockParameters()
    assert math.isclose(
        calculate_neutral_drag_coefficient(param), 0.001886, abs_tol=1e-04
    )


# Test cases for the 'calculate_richardson_number' function
def test_calculate_richardson_number():
    param = MockParameters()
    delta_T = 5.0
    U = 10.0
    assert math.isclose(
        calculate_richardson_number(param, delta_T, U), 0.1635, abs_tol=1e-04
    )


# Test cases for the 'calculate_stability_function' function
def test_calculate_stability_function_short_tail():
    param = MockParameters()
    delta_T = 5.0
    U = 10.0
    assert math.isclose(
        calculate_stability_function(param, delta_T, U), 0.9676, abs_tol=1e-04
    )


def test_calculate_stability_function_long_tail():
    param = MockParameters()
    param.stab_func_type = "long_tail"
    delta_T = 5.0
    U = 10.0
    assert math.isclose(
        calculate_stability_function(param, delta_T, U), 0.9678, abs_tol=1e-04
    )


def test_calculate_stability_function_invalid_type():
    param = MockParameters()
    param.stab_func_type = "invalid"
    delta_T = 5.0
    U = 10.0
    with pytest.raises(SystemExit):
        calculate_stability_function(param, delta_T, U)


# Test cases for the 'solve_ODE' function
def test_solve_ODE():
    param = MockParameters()
    result = solve_ODE(param)
    assert result.success
