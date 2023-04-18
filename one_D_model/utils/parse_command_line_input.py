import argparse

from one_D_model import __version__


def command_line_parser():
    """parse command line input"""
    # Parser for command line options
    parser = argparse.ArgumentParser(description="Run the energy balance model.")
    # Add arguments
    parser.add_argument("-pl", "--plot", help="Make potential, bifurcation, ... plots.", action="store_true", default=False)
    parser.add_argument('-V', '--version', action='version', version=f'ABL energy balance model version: {__version__}')
    parser.add_argument('-f', '--function', help='Randomize the model function itself.', action="store_true",
                        default=False)
    parser.add_argument('-sf', '--stab_function', help='Randomize the stability function.', action="store_true",
                        default=False)
    parser.add_argument('-sfmn', '--stab_function_multi_noise', help='Randomize the stability function.',
                        action="store_true", default=False)
    parser.add_argument('-qi', '--Qi', help='Randomize the model parameter Qi.', action="store_true",
                        default=False)
    parser.add_argument('-l', '--Lambda', help='Randomize the model parameter lambda.', action="store_true",
                        default=False)
    parser.add_argument('-u', '--u', help='Randomize the model parameter u.', action="store_true",
                        default=False)
    parser.add_argument('-a', '--all', help='Run model with all randomizations.', action="store_true",
                        default=False)

    return parser.parse_args()


def read_command_line_input():
    # Read command line input
    args = command_line_parser()

    if args.all:
        function = True
        stab_function = True
        Qi = True
        Lambda = True
        u = True
    else:
        function = args.function
        stab_function = args.stab_function
        stab_function_multi_noise = args.stab_function_multi_noise
        Qi = args.Qi
        Lambda = args.Lambda
        u = args.u

    return function, stab_function, Qi, Lambda, u, args.plot, stab_function_multi_noise
