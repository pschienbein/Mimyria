#!/bin/env python3

import yaml
import argparse
import random
import sys

from ase.io import iread, write

from mimyria.autotrain import state_yaml_create_empty_cycle
import mimyria.args as common_args


def main(argv=None):
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Setup a new autotrain process')
    parser.add_argument('--trajectories', nargs='+',  required=True, help='List of trajectories to base the training on')

    parser.add_argument('--velocities',
                        nargs='+',
                        default=[],
                        help='List of trajectory files providing velocities needed to compute spectra. NOTE1: The order must be the same as in --trajectories! NOTE2: Can be the same files as in --trajectories, if positions and velocities are stored in the same file (optional)')

    parser.add_argument('--velocities_property_string',
                        nargs='+',
                        default=[],
                        help='Provide one property string for all velocity files (or for each velocity file individually). This is necessary if the velocity files do not carry any declaration in the comment line and tells the code how to interpret the columns provided. For instance: "Properties=species:S:1:vel:R:3" (optional)')

    common_args.cell(parser)
    common_args.target(parser)

    parser.add_argument('--mode',
                        type=str,
                        choices=['single-network'],
                        default='single-network',
                        required=False,
                        help='Training mode. Currently only "single-network" is available which trains on randomly sampled data until the spectrum is self-consistently converged')

    parser.add_argument('--test_set_size',
                        type=int,
                        default=100,
                        help="Number of configurations in the test set (default: 100)")

    parser.add_argument('--sn_batchsize',
                        type=int,
                        default=10,
                        help='Number of configurations to be computed within one training cycle (default: 10)')

    parser.add_argument('--sn_max', type=int, default=1000, help='Number of frames to randomly extract from the trajectories; depending on when the spectrum is self-consistently converged, not all are used')

    parser.add_argument('--max_cycles', type=int, default=20,
                        help='Maximum number of cycles (iterations) to perform')

    parser.add_argument('--predict_batchsize', type=int, default=8,
                        help='When predicting: The size of batches processed in parallel on the GPU. More is usually faster, but requires more GPU memory')

    args = parser.parse_args(argv)


    # load state
    state = None
    try:
        with open('state.yaml') as f:
            state = yaml.safe_load(f)

        print('state.yaml is already existing, exiting...')
        print(state)

    except:
        pass

    if state is not None:
        exit()

    state = dict()

    # store the trajectory file names
    state['trajectories'] = args.trajectories

    if len(args.velocities) > 0:
        if len(args.velocities) != len(args.trajectories):
            print('Number of trajectory files containing velocities (--velocities) must be equal to number of trajectory files containing positions (--trajectories)', file=sys.stderr)
            exit(1)

        state['velocities'] = args.velocities

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

    state['mode'] = args.mode
    state['target'] = args.target
    state['backend'] = 'cp2k'
    state['test_set_size'] = args.test_set_size
    state['use_submit_decorator_cpu'] = 1
    state['use_submit_decorator_gpu'] = 1
    state['max_cycles'] = args.max_cycles
    state['cycles'] = []
    first_cycle = {
            'testset': {'status': 'pending'},
            **state_yaml_create_empty_cycle()
            }
    state['cycles'].append(first_cycle)

    # state['cell'] = np.array(args.cell).reshape(3, 3).tolist()
    cell_loader = common_args.CellLoader(args)
    cells = []

    print('# NOTE: Currently cell definitions are always overridden '
          'by the given command line arguments')

    # single-network
    if state['mode'] == 'single-network':
        # complete the state dict
        state['sn'] = dict()
        state['sn']['batchsize'] = args.sn_batchsize
        state['sn']['maxconf'] = args.sn_max

        print('# Random selection of data for training, screening all trajectories which might take a while ....')
        N = args.sn_max + args.test_set_size
        extracted_frames = []
        total_seen = 0
        for traj_file in state['trajectories']:
            cell = cell_loader.get()
            cells.append(cell.tolist())

            for atoms in iread(traj_file):
                total_seen += 1
                if len(extracted_frames) < N:
                    atoms.set_cell(cell)
                    extracted_frames.append(atoms.copy())
                else:
                    r = random.randint(0, total_seen - 1)
                    if r < N:
                        atoms.set_cell(cell)
                        extracted_frames[r] = atoms.copy()

            cell_loader.next_traj()

        # NOTE the extracted configurations are shuffled when using this sampling scheme(this is intended)
        write('extracted-configurations.xyz', extracted_frames, format='extxyz') 

    if cell_loader.bPerTraj:
        state['cell'] = cells
    else:
        state['cell'] = cells[0]

    # safe state
    with open('state.yaml', 'w') as f:
        yaml.safe_dump(state, f)


    print("""Setup of autotrain finished
    NOTE:
    All settings are stored in state.yaml and can be manually adjusted if needed.
    NOTE2:
    Make sure that the run-cpu.sub and run-gpu.sub submission scripts are adapted to the HPC system.
    """)


if __name__ == "__main__":
    main()
