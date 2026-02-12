#!/bin/env python3

import argparse
from ase.io import iread, write

import numpy as np

from mimyria.postprocess import calculate_pgt_from_displacement
from mimyria.io import pgt_flatten


def main(argv=None):
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Script trains a new committee APT NN on given training data')
    parser.add_argument('--configs', type=str, required=True, help='Trajectory file containing the non-displaced atoms')
    parser.add_argument('--polar', type=str, required=True, help='File containing polarizabilities of the displaced atoms')
    parser.add_argument('--displacement', type=float, default=0.01, help='Displacement for the numerical derivative (default: 0.01 Angstrom)', required=False)
    parser.add_argument('--out', type=str, default='pgt.xyz', help='Output file where to write the PGTs (default: pgt.xyz)')
    parser.add_argument('--atoms', nargs='+', type=int, default=None, help='Specify list of atom indices which have been displaced. If not given it is assumed that all atoms have been displaced')

    args = parser.parse_args(argv)

    ###############################################

    #print(f'NOTE: Assuming the displacement {args.displacement} is given in units of Angstrom, converting to Bohr')
    #displacement = args.displacement * 1.88973

    with open(args.out, 'w') as fout:
        pol_traj = iread(args.polar)
        for iFrame, pos_frame in enumerate(iread(args.configs)):
            print(f'Processing frame {iFrame} ...')

            if args.atoms is None:
                atom_indices = range(len(pos_frame))
            else:
                atom_indices = args.atoms

            alphas = []
            for i in atom_indices:
                atom_alphas = []
                for j in range(6):
                    atom_alphas.append(next(pol_traj))

                alphas.append(atom_alphas)

            # NOTE
            # this gives the PGT in e^2 a_0^2 / (Eh * Angstrom)
            # This convention is picked since when multiplying by the velocity
            # (Angstrom / fs), alpha is directly obtained in atomic units.
            pgts = calculate_pgt_from_displacement(alphas, args.displacement)

            if args.atoms is None:
                pos_frame.arrays['pgt'] = pgt_flatten(pgts)
            else:
                tmp = np.zeros((len(pos_frame), 3, 3, 3))
                for i, j in enumerate(atom_indices):
                    tmp[j] = pgts[i]
                pos_frame.arrays['pgt'] = pgt_flatten(tmp)

            write(fout, pos_frame, format='extxyz')

        # Both trajectories should be at their end of file by now
        # otherwise it's likely that there was an error
        eof = False
        try:
            tmp = next(pol_traj)
        except StopIteration:
            # this is the desired outcome
            pass
        else:
            print(f'WARNING: EOF of {args.polar} not reached and thus contains too many frames; this likely is an indication of a mismatch between {args.configs} and {args.polar}, which might result in wrong PGTs!')


if __name__ == "__main__":
    main()
