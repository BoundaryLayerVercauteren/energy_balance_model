import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
from dataclasses_json import dataclass_json

from main import create_u_range, run_sde_model, save_parameters_in_file
from one_D_model.model import parameters


# Helper function to create temporary directories and cleanup after tests
@pytest.fixture(scope="function")
def temp_directory(request):
    temp_dir = tempfile.mkdtemp()
    yield temp_dir


def test_save_parameters_in_file(temp_directory):
    # Mock the parameter class
    @dataclass_json
    @dataclass
    class MockParamClass:
        param1: int = 1
        param2: int = 2
        sol_directory_path: str = temp_directory + "/"

    param_vals = MockParamClass()

    # Call the function to be tested
    save_parameters_in_file(param_vals)

    # Check if the file was created and contains the correct content
    file_name = os.path.join(temp_directory, "parameters.json")
    assert os.path.exists(file_name)
    with open(file_name, "r") as file:
        data = json.load(file)

    assert data == {
        "param1": 1,
        "param2": 2,
        "sol_directory_path": temp_directory + "/",
    }


def test_create_u_range():
    # Call the function with some input
    param_vals = MagicMock()
    param_vals.u_range_start = 0
    param_vals.u_range_end = 10
    param_vals.num_steps = 12
    u_range = create_u_range(param_vals, num_u_steps=6)

    # Check if the output is as expected
    expected_u_range = np.array(
        [0.0, 0.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 10.0, 10.0]
    )
    assert np.allclose(u_range, expected_u_range)


def test_run_sde_model_successful_run():
    # Load Parameters
    param = parameters.Parameters()
    param.num_simulation = 2
    param.num_proc = 1
    # Make directory for output
    param.sol_directory_path = "output/test/"
    if not os.path.exists(param.sol_directory_path):
        os.makedirs(param.sol_directory_path)
        os.makedirs(param.sol_directory_path + "temporary/")
    # Test if model can be run for all randomization types
    try:
        run_sde_model(
            param, function=True, wind=True, wind_and_function=True, stab_function=True
        )
    except Exception as exc:
        assert False, f"Model run raised an exception: {exc}"
    # Remove output directory
    shutil.rmtree(param.sol_directory_path)
