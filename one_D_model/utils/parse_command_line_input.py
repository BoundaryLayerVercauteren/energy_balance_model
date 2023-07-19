import argparse

from one_D_model import __version__


def command_line_parser():
    """Parse command line input"""
    # Parser for command line options
    parser = argparse.ArgumentParser(description="Run the energy balance model.")
    # Add arguments
    parser.add_argument("-pl", "--plot", help="Make potential, bifurcation, ... plots.", action="store_true", default=False)
    parser.add_argument('-V', '--version', action='version', version=f'ABL energy balance model version: {__version__}')
    parser.add_argument('-odeu', '--ode_with_var_u',
                        help='Run ODE model with time dependent u.', action="store_true",
                        default=False)
    parser.add_argument('-f', '--function', help='Solve the model with additive noise (eq. 3).', action="store_true",
                        default=False)
    parser.add_argument('-u', '--u', help='Solve the model with a randomized wind forcing (eq. 4).',
                        action="store_true", default=False)
    parser.add_argument('-uf', '--u_and_function', help='Solve the model with a randomized wind forcing and additive '
                                                        'noise (eq. 5).',  action="store_true", default=False)
    parser.add_argument('-sf', '--stab_function', help='Solve the model with a randomized stability function (eq. 6).',
                        action="store_true", default=False)
    parser.add_argument('-sfu', '--stab_function_and_time_dependent_u',
                        help='Solve the model with a randomized stability function and time-varying wind forcing '
                             '(figure 10).', action="store_true", default=False)
    parser.add_argument('-ss', '--sensitivity_study',
                        help='Perform a sensitivity study (e.g..', action="store_true",
                        default=False)
    parser.add_argument('-a', '--all', help='Run model with all randomizations (eq. 3-6).', action="store_true",
                        default=False)

    return parser.parse_args()


def read_command_line_input():
    # Read command line input
    args = command_line_parser()
    # Set all to true when flag is 'all'
    if args.all:
        function = True
        stab_function = True
        u = True
        uf = True
    else:
        function = args.function
        stab_function = args.stab_function
        u = args.u
        uf = args.u_and_function

    return function, stab_function, u, args.plot, uf, args.stab_function_and_time_dependent_u, args.sensitivity_study, \
           args.ode_with_var_u