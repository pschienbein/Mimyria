#!/bin/env python3

import copy
import argparse

from ase.io import iread, write


def main(argv=None):
    parser = argparse.ArgumentParser(description='Takes a trajectory XYZ file and displaces the respective atom identified with "atom_id" in the comment line')
    parser.add_argument('--infile', type=str, help='<Required> XYZ file containing the configurations', required=True)
    parser.add_argument('--outfile', type=str, help='File where the displaced configurations are printed (default: displaced.xyz)', default='displaced.xyz')
    parser.add_argument('--displacement', type=float, help='Displacement of the atoms for the numerical derivative in angstrom (default: 0.01)', default=0.01)
    parser.add_argument('--atoms', nargs='+', type=int, default=None, help='Specify list of atom indices to displace. If not given all atoms are displaced')

    args = parser.parse_args(argv)

    d = args.displacement
    d_vectors = [
            [d, 0, 0], [-d, 0, 0],
            [0, d, 0], [0, -d, 0],
            [0, 0, d], [0, 0, -d]
            ]

    with open(args.outfile, 'w') as fout:
        for atoms in iread(args.infile):

            if args.atoms is None:
                atom_indices = range(len(atoms))
            else:
                atom_indices = args.atoms

            for iAtom in atom_indices:
                for vd in d_vectors:
                    displaced = copy.deepcopy(atoms)
                    displaced.info['_atom_id'] = iAtom
                    displaced.positions[iAtom] += vd
                    write(fout, displaced, format='extxyz')


if __name__ == "__main__":
    main()
