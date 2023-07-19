import pytest
import math
import numpy as np
from one_D_model.model.solve_ODE import (
    calculate_neutral_drag_coefficient,
    calculate_richardson_number,
    calculate_stability_function,
    define_ODE,
    solve_ODE,
    solve_ODE_with_time_dependent_u,
    calculate_potential,
)


# Define a mock parameter class for testing purposes
class MockParameters:
    kappa = 0.4
    zr = 100.0
    z0 = 0.01
    grav = 9.81
    Tr = 300.0
    stab_func_type = 'short_tail'
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
    assert math.isclose(calculate_neutral_drag_coefficient(param), 0.001886, abs_tol=1e-06)


# Test cases for the 'calculate_richardson_number' function
def test_calculate_richardson_number():
    param = MockParameters()
    delta_T = 5.0
    U = 10.0
    assert math.isclose(calculate_richardson_number(param, delta_T, U), 0.1635, abs_tol=1e-06)


# Test cases for the 'calculate_stability_function' function
def test_calculate_stability_function_short_tail():
    param = MockParameters()
    delta_T = 5.0
    U = 10.0
    assert calculate_stability_function(param, delta_T, U) == pytest.approx(0.9801986733067555)


def test_calculate_stability_function_long_tail():
    param = MockParameters()
    param.stab_func_type = 'long_tail'
    delta_T = 5.0
    U = 10.0
    assert calculate_stability_function(param, delta_T, U) == pytest.approx(0.9801986733067555)


def test_calculate_stability_function_invalid_type():
    param = MockParameters()
    param.stab_func_type = 'invalid'
    delta_T = 5.0
    U = 10.0
    with pytest.raises(SystemExit):
        calculate_stability_function(param, delta_T, U)


# Test cases for the 'define_ODE' function
def test_define_ODE():
    param = MockParameters()
    t = 0.0
    delta_T = 5.0
    u = 10.0
    Lambda = 0.01
    Qi = 100.0
    expected_result = pytest.approx(-0.09310838023522712)
    assert define_ODE(t, delta_T, u, Lambda, Qi, param) == expected_result


# Test cases for the 'solve_ODE' function
def test_solve_ODE():
    param = MockParameters()
    result = solve_ODE(param)
    assert result.success


def test_solve_ODE_with_time_dependent_u():
    param = MockParameters()
    result = solve_ODE_with_time_dependent_u(param)
    assert result.shape == param.t_span.shape


# Test cases for the 'calculate_potential' function
def test_calculate_potential():
    param = MockParameters()
    delta_T = np.linspace(0, 5, 11)
    u = 10.0
    potential = calculate_potential(delta_T, u, param)
    expected_result = np.array([0.0, 0.175, 0.353, 0.533, 0.714, 0.896, 1.080, 1.265, 1.452, 1.640, 1.830])
    assert np.allclose(potential, expected_result, atol=1e-3)
