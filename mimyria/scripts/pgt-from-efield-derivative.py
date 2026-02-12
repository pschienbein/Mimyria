#!/bin/env python3

import argparse
from ase.io import iread, write

from mimyria.postprocess import calculate_pgt_from_field
from mimyria.io import pgt_flatten


def main(argv=None):
    displacements = ['0',
                     'px', 'mx', 'py', 'my', 'pz', 'mz',
                     'pxpy', 'pxpz', 'pypz', 'mxmy', 'mxmz', 'mymz']

    # Command line argument parser
    parser = argparse.ArgumentParser(description='Script trains a new committee APT NN on given training data')
    parser.add_argument('--pos', type=str, required=True)
    parser.add_argument('--ftemplate', type=str, help='Allows to give a single file template to specify all efield files, for instance "field_{disp}-dft-frc-1.xyz"')
    parser.add_argument('--field_strength', type=float, default=5e-4, help='Strength of the applied electric field (default: 5e-4 atomic units)', required=False)
    parser.add_argument("--field_mode", choices=["component", "normalized"], default="component", help=(
        "How field_strength is applied. "
        "'component': apply field_strength to each Cartesian component (default). "
        "'normalized': normalize the field direction first, so the total field "
        "magnitude equals field_strength."
    )
)
    parser.add_argument('--out', type=str, required=True)

    for d in displacements:
        dx = d.replace('p', '+').replace('m', '-')
        parser.add_argument(f'--f{d}', type=str, default=f'field_{d}-dft-frc-1.xyz',
                            help=f'Forces for efield in {dx} direction (default: field_{d}-dft-frc-1.xyz')

    args = parser.parse_args(argv)

    normalized_field = args.field_mode == 'normalized'

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

        # NOTE
        # this gives the PGT in e^2 a_0 / (Eh)
        # convert to e^2 a_0^2 / (Eh * Angstrom) instead by convention,
        # since when multiplying by the velocity
        # (Angstrom / fs), alpha is directly obtained in atomic units.
        pgts = calculate_pgt_from_field(atoms, args.field_strength, normalized_field) * 1.88973
        print(pgts.shape)

        trajectory = atoms['pos']
        trajectory.arrays['pgt'] = pgt_flatten(pgts)
        frames2write.append(trajectory)

        print(f'Processed frame {step}')

    write(args.out, frames2write)


if __name__ == "__main__":
    main()
