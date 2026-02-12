#!/bin/env python3

import os
import yaml
from pathlib import Path


def main(argv=None):
    # script can be called from one of the sub directories
    cwd = Path.cwd()
    current_dir = cwd.name

    subdirs = ["refdata", 'training']
    if current_dir in subdirs:
        os.chdir(cwd.parents[1])
        print(f'changed working directory to: {Path.cwd()}')

    # load state
    state = None
    try:
        with open('state.yaml') as f:
            state = yaml.safe_load(f)
    except:
        print('No state.yaml found, use autotrain_setup.py first to create an autotrain process')
        exit()


    # get latest step
    mode = state['mode']

    if mode == 'single-network':
        from mimyria.autotrain_single_network import refdata, training, testset

        function = {
                'testset': testset,
                'refdata': refdata,
                'training': training
                }

        # defined order, refdata < training
        bExit = False
        while not bExit:
            latest = state['cycles'][-1]
            for step in ['testset', 'refdata', 'training']:
                if step in latest:
                    if latest[step]['status'] != 'done':
                        if not function[step](state):
                            bExit = True
                            break
                        else:
                            # update state
                            with open('state.yaml', 'w') as f:
                                yaml.safe_dump(state, f)

    # update state
    with open('state.yaml', 'w') as f:
        yaml.safe_dump(state, f)


if __name__ == "__main__":
    main()
