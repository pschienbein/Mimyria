#!/bin/env python3

import os
import yaml
import argparse
import sys
import numpy as np

from mimyria.autotrain import submit_job


def main(argv=None):
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Uses the computed APTs (or PGTs) to compute IR (or Raman) spectra during the autotrain process for the given training cycle')
    parser.add_argument('--cycle',
                        type=int,
                        required=True,
                        help='The Autotrain cycle for which the full prediction is to be carried out')

    parser.add_argument('--velocities', 
                        nargs='+',  
                        default=[], 
                        help='List of trajectory files providing velocities needed to compute spectra. NOTE1: The order must be the same as in --trajectories! NOTE2: Can be the same files as in --trajectories, if positions and velocities are stored in the same file (optional)')

    parser.add_argument('--velocities_property_string', 
                        nargs='+',  
                        default=[], 
                        help='Provide one property string for all velocity files (or for each velocity file individually). This is necessary if the velocity files do not carry any declaration in the comment line and tells the code how to interpret the columns provided. For instance: "Properties=species:S:1:vel:R:3" (optional)')

    parser.add_argument('--force_local_execution',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Does NOT submit the job to the queue, but instead runs the spectrum calculation on the current node (default: False)')

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

    # test if there are velocities: 
    if 'velocities' not in state:
        if len(args.velocities) == 0:
            print('No velocities provided!', file=sys.stderr)
            print('NOTE: To compute spectra, velocities need to be available for each MD trajectory. Those can be set via autotrain_setup.py OR the --velocities argument of this script!', file=sys.stderr)
            exit(1)

        # velocities must be set
        state['velocities'] = args.velocities

        if len(state['velocities']) != len(state['trajectories']):
            print('Number of trajectory files containing velocities (--velocities) must be equal to number of trajectory files containing positions (--trajectories)', file=sys.stderr)
            exit(1)

        if len(args.velocities_property_string) > 0:
            if len(args.velocities_property_string) == 1:
                state['velocities_properties'] = []

                for i in range(len(args.velocities)):
                    state['velocities_properties'].append(args.velocities_property_string[0])

            else:
                if len(args.velocities) != len(args.velocities_property_string):
                    print('number of property strings in --velocities_property_string must either be one or equal to the number of trajectory files in --velocities', file=sys.stderr)
                    exit(1)

                state['velocities_properties'] = args.velocities_property_string

    # create spec subdir
    path = f'cycle{cycle_no}/predict_full/spec/'
    os.makedirs(path, exist_ok=True)

    # write the filelist
    filelist = open(path + 'filelist', 'w')
    width = int(np.ceil(np.log10(len(state['trajectories']))))
    for i, fvel in enumerate(state['velocities']):
        rp = os.path.realpath(fvel)
        filelist.write(f'../predict-{i:0{width}d}.xyz.zstd, {rp}')

        if 'velocities_properties' in state:
            props = state['velocities_properties'][i]
            filelist.write(f' ({props})')
        filelist.write('\n')
    filelist.close()

    fout = open(path + 'config.ini', 'w')
    fout.write(f'[{state["target"]}2cf]\n')
    fout.close()

    fout = open(path + 'run.sh', 'w')
    fout.write('mimyria config.ini --filelist filelist\n')
    fout.close()

    ret = submit_job(state, path, 'cpu', bWait=False, bForceLocalExecution=args.force_local_execution)


if __name__ == "__main__":
    main()
