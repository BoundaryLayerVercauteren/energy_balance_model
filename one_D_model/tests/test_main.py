import os
import shutil

import warnings

# Don't show any type of deprecation warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    import main

from one_D_model.model import parameters


def test_successful_run_Qi():
    # Load Parameters
    param = parameters.Parameters()
    param.num_simulation = 2
    param.num_proc = 1
    # Make directory for output
    param.sol_directory_path = 'output/test/'
    if not os.path.exists(param.sol_directory_path):
        os.makedirs(param.sol_directory_path)
        os.makedirs(param.sol_directory_path + 'temporary/')
    # Test if model can be run for stochastic u
    try:
        main.run_model(param, Qi=True)
        shutil.rmtree(param.sol_directory_path)
    except Exception as exc:
        shutil.rmtree(param.sol_directory_path)
        assert False, f"Model run with stochastic Qi raised an exception: {exc}"


def test_successful_run_Lambda():
    # Load Parameters
    param = parameters.Parameters()
    param.num_simulation = 2
    param.num_proc = 1
    # Make directory for output
    param.sol_directory_path = 'output/test/'
    if not os.path.exists(param.sol_directory_path):
        os.makedirs(param.sol_directory_path)
        os.makedirs(param.sol_directory_path + 'temporary/')
    # Test if model can be run for stochastic u
    try:
        main.run_model(param, Lambda=True)
        shutil.rmtree(param.sol_directory_path)
    except Exception as exc:
        shutil.rmtree(param.sol_directory_path)
        assert False, f"Model run with stochastic Lambda raised an exception: {exc}"


def test_successful_run_u():
    # Load Parameters
    param = parameters.Parameters()
    param.num_simulation = 2
    param.num_proc = 1
    # Make directory for output
    param.sol_directory_path = 'output/test/'
    if not os.path.exists(param.sol_directory_path):
        os.makedirs(param.sol_directory_path)
        os.makedirs(param.sol_directory_path + 'temporary/')
    # Test if model can be run for stochastic u
    try:
        main.run_model(param, u=True)
        shutil.rmtree(param.sol_directory_path)
    except Exception as exc:
        shutil.rmtree(param.sol_directory_path)
        assert False, f"Model run with stochastic u raised an exception: {exc}"


def test_successful_run_stab_function():
    # Load Parameters
    param = parameters.Parameters()
    param.num_simulation = 2
    param.num_proc = 1
    # Make directory for output
    param.sol_directory_path = 'output/test/'
    if not os.path.exists(param.sol_directory_path):
        os.makedirs(param.sol_directory_path)
        os.makedirs(param.sol_directory_path + 'temporary/')
    # Test if model can be run for stochastic u
    try:
        main.run_model(param, stab_function=True)
        shutil.rmtree(param.sol_directory_path)
    except Exception as exc:
        shutil.rmtree(param.sol_directory_path)
        assert False, f"Model run with stochastic stab_function raised an exception: {exc}"


def test_successful_run_stab_function_multi_noise():
    # Load Parameters
    param = parameters.Parameters()
    param.num_simulation = 2
    param.num_proc = 1
    # Make directory for output
    param.sol_directory_path = 'output/test/'
    if not os.path.exists(param.sol_directory_path):
        os.makedirs(param.sol_directory_path)
        os.makedirs(param.sol_directory_path + 'temporary/')
    # Test if model can be run for stochastic u
    try:
        main.run_model(param, stab_function_multi_noise=True)
        shutil.rmtree(param.sol_directory_path)
    except Exception as exc:
        shutil.rmtree(param.sol_directory_path)
        assert False, f"Model run with stochastic stab_function_multi_noise raised an exception: {exc}"


def test_successful_run_u_and_function():
    # Load Parameters
    param = parameters.Parameters()
    param.num_simulation = 2
    param.num_proc = 1
    # Make directory for output
    param.sol_directory_path = 'output/test/'
    if not os.path.exists(param.sol_directory_path):
        os.makedirs(param.sol_directory_path)
        os.makedirs(param.sol_directory_path + 'temporary/')
    # Test if model can be run for stochastic u
    try:
        main.run_model(param, u_and_function=True)
        shutil.rmtree(param.sol_directory_path)
    except Exception as exc:
        shutil.rmtree(param.sol_directory_path)
        assert False, f"Model run with stochastic u_and_function raised an exception: {exc}"
