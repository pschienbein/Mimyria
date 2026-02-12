import argparse
import numpy as np


def target(parser, default=None):
    if default is not None:
        parser.add_argument('--target',
                            type=str,
                            choices=['apt', 'pgt'],
                            default=default,
                            help='Property to train or compare')
    else:
        parser.add_argument('--target',
                            type=str,
                            choices=['apt', 'pgt'],
                            required=True,
                            help='Property to train or compare')


def cell(parser, required=True):
    parser.add_argument('--cell',
                        type=float,
                        nargs=9,
                        required=False,
                        help='3x3 matrix describing the cell (only --cell_def global)')

    parser.add_argument('--cellfn',
                        type=str,
                        required=False,
                        help='Filename containing all cells (see --cell_def)')

    parser.add_argument('--cell_def',
                        type=str,
                        choices=['global', 'per-trajectory'],
                        default='global',
                        help='Cell definition the same for all frames (global), '
                             'cell different for each trajectory (per-trajectory)')

    parser.add_argument('--cell_override',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='If true, overrides all cell definitions that '
                             'might be present in the trajectory files')


class CellLoader:
    def __init__(self, args):
        self.args = args
        self.bPerTraj = (args.cell_def == 'per-trajectory')

        if args.cellfn is not None:
            self.fin = open(args.cellfn)
            line = next(self.fin)
            aLine = line.split()
            self.current_cell = np.array([float(x) for x in aLine]).reshape(3, 3)
        elif args.cell is not None:
            self.current_cell = np.array(args.cell).reshape(3, 3)
        else:
            self.current_cell = None

    def get(self):
        if self.current_cell is None:
            raise RuntimeError('No Cell Definition given in the command line '
                               'arguments, but requested from the script')

        return self.current_cell

    def next_traj(self):
        if self.bPerTraj and self.fin is not None:
            try:
                line = next(self.fin)
                aLine = line.split()
                self.current_cell = np.array([float(x) for x in aLine]).reshape(3, 3)

            except StopIteration:
                self.current_cell = None
