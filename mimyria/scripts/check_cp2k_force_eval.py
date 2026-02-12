import argparse

import mimyria.args as common_args
from mimyria.calculator import CP2kBackend


def main(argv=None):
    parser = argparse.ArgumentParser(description='Checks if the given CP2k input file (FORCE_EVAL section) is suitable for the given target calculation')
    parser.add_argument('--cp2k_inp', type=str, required=True)
    common_args.target(parser, default='apt')

    args = parser.parse_args(argv)

    try:
        back = CP2kBackend()
        if args.target == 'apt':
            back.apt_check_force_eval(args.cp2k_inp)
        elif args.target == 'pgt': 
            back.pgt_check_force_eval(args.cp2k_inp)

    except Exception as e:
        print('PROBLEM with the CP2k input:')
        print(e)
