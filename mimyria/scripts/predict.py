#!/bin/env python3

import argparse
import time
import datetime
import sys

import torch

from ase.io import iread, write

from mimyria.models import load_model
import mimyria.args as common_args

from mimyria.io import atoms_arrays_flatten, open_wrapper


def main(argv=None):
    # Start time
    tstart = time.perf_counter()

    # Command line argument parser
    parser = argparse.ArgumentParser(description='Script predicts data given a model and configurations')
    parser.add_argument('--model', type=str, default='model.mym')
    common_args.cell(parser)
    parser.add_argument('--configs', type=str, required=True)
    parser.add_argument('--out', type=str, default='prediction.xyz')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--verbose', action='store_true', default=False, help='Prints additional memory and timing information at each step')
    args = parser.parse_args(argv)

    ###############################################

    cell_loader = common_args.CellLoader(args)

    # cell = np.array(args.cell).reshape(3, 3)
    # print(cell)

    mdl = load_model('cuda:0', args.model)
    timings_sum = 0
    timings_N = 0

    N = args.batch_size

    if args.verbose:
        fout_verbose = open('verbose.log', 'w')
        fout_verbose.write('# Time | Available Memory | Used Memory')

    with open_wrapper(args.configs) as fin, open_wrapper(args.out, 'w') as fout:
        buffer = []

        for atoms in iread(fin):
            if atoms.cell is None or atoms.cell.volume < 1e-8:
                atoms.set_cell(cell_loader.get())

            buffer.append(atoms)

            if len(buffer) == N:
                beg = time.perf_counter()
                mdl.predict(buffer)
                end = time.perf_counter()
                timings_sum += (end - beg) / N
                timings_N += 1

                print(f'{timings_sum / timings_N}\r', file=sys.stderr)

                if args.verbose:
                    mem_used = torch.cuda.max_memory_allocated()
                    current_device = torch.cuda.current_device()
                    mem_avail = torch.cuda.get_device_properties(current_device).total_memory
                    fout_verbose.write(f'{timings_sum / timings_N} {mem_avail / (1024 ** 3):.4f} {mem_used / (1024 ** 3):.4f}\n')
                    fout_verbose.flush()

                for atoms in buffer:
                    atoms_arrays_flatten(atoms)
                    write(fout, atoms, format='extxyz')

                buffer = []

        # handle remaining frames
        if buffer:
            mdl.predict(buffer)

            for atoms in buffer:
                atoms_arrays_flatten(atoms)
                write(fout, atoms, format='extxyz')

    print('')

    print('==============================')
    print('Timing:')
    elapsed = time.perf_counter() - tstart
    print(f'Total run time: {datetime.timedelta(seconds=elapsed)}')
    mean = timings_sum / timings_N
    print(f'Average time per config: {datetime.timedelta(seconds=mean)}')

    mem_used = torch.cuda.max_memory_allocated()
    current_device = torch.cuda.current_device()
    mem_avail = torch.cuda.get_device_properties(current_device).total_memory
    print('==============================')
    print('Memory Statistics:')
    print(f'Available memory: {mem_avail / (1024 ** 3):.2f} GiB')
    print(f'Maximum used: {mem_used / (1024 ** 3):.2f} GiB')
    print(f'Used percentage: {mem_used / mem_avail * 100:.2f} %')


if __name__ == "__main__":
    main()
