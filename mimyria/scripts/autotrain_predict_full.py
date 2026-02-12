#!/bin/env python3

import yaml
import argparse


def main(argv=None):
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Predicts all APT (or PGTs) for all given trajectories specified in the autotrain process')
    parser.add_argument('--cycle',
                        type=int,
                        required=True,
                        help='The Autotrain cycle for which the full prediction is to be carried out')
    parser.add_argument('--continue', dest='cont', action='store_true', default=False, help='Continues a previously stopped predict full step')

    args = parser.parse_args(argv)

    # load state
    state = None
    try:
        with open('state.yaml') as f:
            state = yaml.safe_load(f)
    except:
        print('No state.yaml found, use autotrain_setup.py first to create an autotrain process')
        exit()

    # get cycle number
    cycle_no = args.cycle

    try:
        mode = state['mode']
        if mode == 'single-network':
            from mimyria.autotrain_single_network import predict_full
            predict_full(state, cycle_no, args.cont)

    except FileExistsError as e:
        print(f'ERROR: {e}')
        print( '  HINT: If you want to redo the full calculation, remove the path')
        print( '  HINT: If you want to continue the calculation, add the --continue flag')
        print( '        WARNING: continue does NOT check for the completeness of the prediction, but continues with the first missing trajectory')

    except Exception as e:
        print(f'ERROR: {e}')


if __name__ == "__main__":
    main()
