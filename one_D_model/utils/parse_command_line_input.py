import argparse

from one_D_model import __version__


def command_line_parser():
    """Parse command line input"""
    # Parser for command line options
    parser = argparse.ArgumentParser(description="Run the energy balance model.")
    # Add arguments
    parser.add_argument("-pl", "--plot", help="Make potential, bifurcation, ... plots.", action="store_true", default=False)
    parser.add_argument('-V', '--version', action='version', version=f'ABL energy balance model version: {__version__}')
    parser.add_argument('-f', '--function', help='Randomize the model function itself.', action="store_true",
                        default=False)
    parser.add_argument('-sf', '--stab_function', help='Randomize the stability function.', action="store_true",
                        default=False)
    parser.add_argument('-qi', '--Qi', help='Randomize the model parameter Qi.', action="store_true",
                        default=False)
    parser.add_argument('-l', '--Lambda', help='Randomize the model parameter lambda.', action="store_true",
                        default=False)
    parser.add_argument('-u', '--u', help='Randomize the model parameter u.', action="store_true",
                        default=False)
    parser.add_argument('-uf', '--u_and_function', help='Randomize the model parameter u and the model itself.',
                        action="store_true", default=False)
    parser.add_argument('-sfu', '--stab_function_and_time_dependent_u',
                        help='Randomize the stability function and u is time dependent.', action="store_true",
                        default=False)
    parser.add_argument('-ss', '--sensitivity_study',
                        help='Perform a sensitivity study.', action="store_true",
                        default=False)
    parser.add_argument('-odeu', '--ode_with_var_u',
                        help='Run ODE model with time dependent u.', action="store_true",
                        default=False)
    parser.add_argument('-a', '--all', help='Run model with all randomizations.', action="store_true",
                        default=False)

    return parser.parse_args()


def read_command_line_input():
    # Read command line input
    args = command_line_parser()
    # Sett all to true when flag is 'all'
    if args.all:
        function = True
        stab_function = True
        Qi = True
        Lambda = True
        u = True
        uf = True
        sfu = True
    else:
        function = args.function
        stab_function = args.stab_function
        Qi = args.Qi
        Lambda = args.Lambda
        u = args.u
        uf = args.u_and_function
        sfu = args.stab_function_and_time_dependent_u
        ss = args.sensitivity_study
        odeu = args.ode_with_var_u

    return function, stab_function, Qi, Lambda, u, args.plot, uf, sfu, ss, odeu
