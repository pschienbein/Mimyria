#!/bin/env python3

import argparse
from ase.io import iread, write

import numpy as np

from mimyria.postprocess import calculate_apt_from_displacement
from mimyria.io import apt_flatten
import mimyria.args as common_args


def main(argv=None):
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Script trains a new committee APT NN on given training data')
    parser.add_argument('--configs', type=str, required=True, help='Trajectory file containing the non-displaced atoms')
    parser.add_argument('--wannier', type=str, required=True, help='File containing the Wannier centers of the displaced atoms')
    parser.add_argument('--displacement', type=float, default=0.01, help='Displacement for the numerical derivative (default: 0.01 Angstrom)', required=False)
    parser.add_argument('--out', type=str, default='apt.xyz', help='Output file where to write the APTs (default: apt.xyz)')
    parser.add_argument('--charges', nargs='+', type=str, help='Charges of each atom kind as (examples): "H:1 O:6 ..."', required=True)

    common_args.cell(parser, required=False)

    args = parser.parse_args(argv)

    ###############################################

    cell = None
    if args.cell is not None:
        cell = np.array(args.cell).reshape(3, 3)

    with open(args.out, 'w') as fout:
        wan_traj = iread(args.wannier)
        for iFrame, pos_frame in enumerate(iread(args.configs)):
            print(f'Processing frame {iFrame} ...')

            wanniers = []
            for i in range(len(pos_frame)):
                atom_wanniers = []
                for j in range(6):
                    atom_wanniers.append(next(wan_traj))
                    if cell is not None:
                        atom_wanniers[-1].set_cell(cell)
                        atom_wanniers[-1].set_pbc(True)

                wanniers.append(atom_wanniers)

            apts = calculate_apt_from_displacement(wanniers, args.displacement)
            pos_frame.arrays['apt'] = apt_flatten(apts)
            write(fout, pos_frame, format='extxyz')


if __name__ == "__main__":
    main()
