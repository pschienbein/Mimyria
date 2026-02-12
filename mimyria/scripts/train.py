#!/bin/env python3


import datetime
import argparse
import time

import numpy as np
import torch

from ase.io import read

from mimyria.models import model_from_target
from mimyria.models.parameters import enum_type, EDataNormalization, get_default
import mimyria.args as common_args


def main(argv=None):
    # Start time
    tstart = time.perf_counter()

    # Command line argument parser
    parser = argparse.ArgumentParser(description='Script trains a new APTNN or PGTNN. Note that there are different defaults for the different networks. The architecture arguments likely do not need to be adjusted')
    common_args.target(parser)
    parser.add_argument('--train', type=str, required=True)
    common_args.cell(parser)
    parser.add_argument('--out', type=str, default='model.mym')
    parser.add_argument('--learning_curve', type=str, default='learning-curve.dat', help='File where the learning curve is to be stored (default: learning-curve.dat)')
    parser.add_argument('--learning_curve_irreps', type=str, default=None, help='If set, file where the learning curve per irrep block is printed (default: disabled)')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of maximum epochs to train (default: 500). Note that there is a scheduler active which might stop the training earlier')
    parser.add_argument('--fix_seed', action=argparse.BooleanOptionalAction, default=False, help='Fixes the random seed')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of batched processed in parallel (default: 1). Increase depending on your available memory (does also affect training, thus RMSEs change)')
    parser.add_argument('--initial_learning_rate', type=float, default=1e-2, help='Initial learning rate (default: 1e-2). Intentially set rather large, decrease if necessary (does also affect training, thus RMSEs change)')
    parser.add_argument('--num_features', type=int, help='Architecture: Hidden layer multiplicity')
    parser.add_argument('--num_layers', type=int, help='Architecture: Number of hidden layers')
    parser.add_argument('--lmax', type=int, help='Architecture: Maximum l')
    parser.add_argument('--radial_cutoff', type=int, help='Architecture: Radial Cutoff to construct the graph')
    parser.add_argument('--num_radial_basis', type=int, help='Architecture: Number of radial basis functions')
    parser.add_argument('--normalization', type=enum_type(EDataNormalization), choices=list(EDataNormalization), help='Architecture: Data normalization')
    parser.add_argument('--equal_num_features_per_channel', action=argparse.BooleanOptionalAction, default=False, help='Architecture: Change the number of features dynamically per l-value')
    parser.add_argument('--natural_parities_only', action=argparse.BooleanOptionalAction, default=False, help='Architecture: Only add natural parities in the hidden layers')

    mdl_param_default_keys = [
            'num_features',
            'num_layers',
            'lmax',
            'radial_cutoff',
            'num_radial_basis',
            'normalization',
            'equal_num_features_per_channel',
            'natural_parities_only'
            ]

    # read knowns:
    args, remaining = parser.parse_known_args()
    param = get_default(args.target)

    # set the defaults
    defaults = dict()
    for key in mdl_param_default_keys:
        defaults[key] = getattr(param, key)
    parser.set_defaults(**defaults)

    # full parse
    args = parser.parse_args(argv)

    ###############################################

    if args.fix_seed:
        # fix the seed (train/test split)
        np.random.seed(0)
        torch.manual_seed(0)

    cell_loader = common_args.CellLoader(args)

    # cell = np.array(args.cell).reshape(3, 3)
    # print(cell)

    atom_kinds = set()
    tdata = read(args.train, ':')
    for atoms in tdata:
        if atoms.cell is None or atoms.cell.volume < 1e-8:
            atoms.set_cell(cell_loader.get())

        atom_kinds.update(atoms.get_chemical_symbols())

    print('Detected atom symbols: ', atom_kinds)

    # param = ModelParameters(atom_kinds=list(atom_kinds))
    param = get_default(args.target, atom_kinds=list(atom_kinds))

    # copy params over from the argparse
    for key in mdl_param_default_keys:
        setattr(param, key, getattr(args, key))

    # create the model
    mdl = model_from_target(args.target, device='cuda:0', model_parameters=param)

    # train it
    mdl.train(tdata,
              num_epochs=args.num_epochs,
              learning_curve_fout=args.learning_curve,
              batch_size=args.batch_size,
              initial_learning_rate=args.initial_learning_rate,
              learning_curve_per_irrep_fout=args.learning_curve_irreps)

    # store the model
    mdl.save(args.out)

    # print timing statistics
    print('==============================')
    print('Timing:')
    elapsed = time.perf_counter() - tstart
    print(f'Total run time: {datetime.timedelta(seconds=elapsed)}')

    # print some memory usage information
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
