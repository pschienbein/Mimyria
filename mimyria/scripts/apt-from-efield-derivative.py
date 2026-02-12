#!/bin/env python3

import argparse
from ase.io import iread, write

from mimyria.postprocess import calculate_apt_from_field


def main(argv=None):
    displacements = ['px', 'mx', 'py', 'my', 'pz', 'mz']

    # Command line argument parser
    parser = argparse.ArgumentParser(description='Script trains a new committee APT NN on given training data')
    parser.add_argument('--pos', type=str, required=True)
    parser.add_argument('--ftemplate', type=str, help='Allows to give a single file template to specify all efield files, for instance "field_{disp}-frc-1.xyz"')
    parser.add_argument('--field_strength', type=float, default=5e-4, help='Strength of the applied electric field (default: 5e-4 atomic units)', required=False)
    parser.add_argument('--out', type=str, required=True)

    for d in displacements:
        dx = d.replace('p', '+').replace('m', '-')
        parser.add_argument(f'--f{d}', type=str, default=f'field_{d}-frc-1.xyz',
                            help=f'Forces for efield in {dx} direction (default: field_{d}-frc-1.xyz')

    args = parser.parse_args(argv)

    ###############################################

    files = {'pos': args.pos}
    if args.ftemplate is not None:
        for d in displacements:
            files[d] = args.ftemplate.format(disp=d)

    else:
        for d in displacements:
            files[d] = getattr(args, f'f{d}')

    iterators = {key: iread(path) for key, path in files.items()}

    frames2write = []
    for step, frames in enumerate(zip(*iterators.values())):
        atoms = dict(zip(iterators.keys(), frames))

        apts = calculate_apt_from_field(atoms, args.field_strength)
        print(apts.shape)

        trajectory = atoms['pos']
        # to store the data in the xyz file
        # NOTE: this is row-major!
        trajectory.arrays['apt'] = apts.reshape(len(apts), 9)
        frames2write.append(trajectory)

        print(f'Processed frame {step}')

    write(args.out, frames2write)


if __name__ == "__main__":
    main()
