#!/bin/env python3

import argparse
from collections import defaultdict
from itertools import zip_longest
import sys

import numpy as np
from ase.io import iread

import mimyria.args as common_args
from mimyria.io import atoms_arrays_reshape, open_wrapper


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare two files containing APTs and provide the rmse')
    parser.add_argument('--confA', type=str, required=True)
    parser.add_argument('--confB', type=str, required=True)
    common_args.target(parser, default='apt')
    parser.add_argument('--out', type=str, default=None, help='Output file for Bias, RMSE, and normalized RMSE (default: STDOUT)')
    parser.add_argument('--scatterout', type=str, default='scatter-{symbol}.dat', help='Output file name, comparing the DFT and NN APTs componentwise (default: scatter-{symbol}.dat)')
    parser.add_argument('--ignore_zeros', action='store_true', default=False, help='Ignore all APTs/PGTs which are strictly zero; helpful when not all atoms have been displaced for benchmarking')

    args = parser.parse_args(argv)

    if args.target == 'dipole':
        raise NotImplementedError

    # overwrite stdout if requested
    if args.out is not None:
        sys.stdout = open(args.out, 'w')

    target = args.target

    print(f'Comparing {target}', file=sys.stderr)

    with open_wrapper(args.confA) as finA, open_wrapper(args.confB) as finB:
        iterA = iread(finA, format=getattr(finA, 'ase_format'))
        iterB = iread(finB, format=getattr(finB, 'ase_format'))

        species_mses = defaultdict(list)
        species_propB_ms = defaultdict(list)
        species_num_outliers = defaultdict(int)
        species_count = defaultdict(int)
        scatter = defaultdict(list)

        component_se = dict()
        component_count = defaultdict(int)

        mses = []
        dens = []

        tensorsA = defaultdict(list)
        tensorsB = defaultdict(list)

        for atomsA, atomsB in zip_longest(iterA, iterB):
            if atomsA is None or atomsB is None:
                print('# the two files have different lengths!', file=sys.stderr)
                exit(1)

            N = len(atomsA)
            if N != len(atomsB):
                print('# mismatch in atom numbers in the two files', file=sys.stderr)
                exit(1)

            atoms_arrays_reshape(atomsA)
            atoms_arrays_reshape(atomsB)

            propA = atomsA.arrays[target]
            propB = atomsB.arrays[target]

            if args.ignore_zeros:
                maskA = np.all(propA == 0, axis=tuple(range(1, propA.ndim)))
                maskB = np.all(propB == 0, axis=tuple(range(1, propB.ndim)))
                mask_keep = ~(maskA | maskB)

                propA = propA[mask_keep]
                propB = propB[mask_keep]
                atomsA = atomsA[mask_keep]
                atomsB = atomsB[mask_keep]

                print(mask_keep)

            for i, symbol in enumerate(atomsA.get_chemical_symbols()):
                tensorsA[symbol].append(propA[i])
                tensorsB[symbol].append(propB[i])

                vA = propA[i].flatten()
                vB = propB[i].flatten()
                for a, b in zip(vA, vB):
                    scatter[symbol].append([a, b])

        # compute metrics
        metrics = defaultdict(dict)

        for sym in tensorsA:
            A = np.array(tensorsA[sym])
            B = np.array(tensorsB[sym])
            diff = A - B

            # use standard deviation
            std = np.std(B, axis=0)
            # check if zero
            if np.any(std < 1e-14):
                print('# [warn]: Standard deviation is close to zero!', file=sys.stderr)

            # BIAS
            bias = np.mean(diff, axis=0)
            metrics[sym]['bias'] = bias
            metrics[sym]['rel_bias'] = bias / std

            # RMSE
            rmse = np.sqrt(np.mean(diff**2, axis=0))
            metrics[sym]['rmse'] = rmse
            metrics[sym]['rel_rmse'] = rmse / std

        # print all
        # format helpers
        def fmt_comp(idx):
            return "[" + ",".join(map(str, idx)) + "]"

        header = ("# Columns: symbol, component, n, bias, rel_bias, rmse, rel_rmse\n"
                  "# bias, rmse in property units; rel_bias, rel_rmse normalized by standard deviation of --confB\n"
                  "# 'rel' values: smaller is better (0 = perfect).\n"
                  "# 'R2' values: Coefficient of Determination (1 = perfect).\n"
                  f"# --confA {args.confA}\n"
                  f"# --confB {args.confB}")
        print(header)
        thead = (f"# {'sym':>4s} {'comp':>8s} {'bias':>12s} {'rel_bias':>12s} "
                 f"{'rmse':>12s} {'rel_rmse':>12s} {'R2':>12s}")

        # collect per-symbol summaries
        results_s = []

        sym_bias_mean = dict()
        sym_rel_bias_mean = dict()
        sym_rmse_mean = dict()
        sym_rel_rmse_mean = dict()
        sym_R2 = dict()

        print('\n# Per-Component, per-symbol averages:')
        print(thead)
        for sym in sorted(metrics.keys()):
            arr_bias = np.asarray(metrics[sym]['bias'])
            arr_rel_bias = np.asarray(metrics[sym]['rel_bias'])
            arr_rmse = np.asarray(metrics[sym]['rmse'])
            arr_rel_rmse = np.asarray(metrics[sym]['rel_rmse'])
            arr_R2 = []

            shape = arr_bias.shape
            for comp in np.ndindex(shape):
                b = float(arr_bias[comp])
                rb = float(arr_rel_bias[comp])
                r = float(arr_rmse[comp])
                rr = float(arr_rel_rmse[comp])
                R2 = 1 - rr**2
                arr_R2.append(R2)
                print(f"{sym:>6s} {fmt_comp(comp):>8s} "
                      f"{b:12.6f} {rb:12.6f} {r:12.6f} {rr:12.6f} {R2:12.6f}")

            # symbol averages (macro mean over components)
            sym_bias_mean[sym] = np.mean(arr_bias)
            sym_rel_bias_mean[sym] = np.mean(arr_rel_bias)
            sym_rmse_mean[sym] = np.mean(arr_rmse)
            sym_rel_rmse_mean[sym] = np.mean(arr_rel_rmse)
            sym_R2[sym] = np.mean(arr_R2)

        print('\n# Per-symbol averages:')
        print(thead)
        for sym in sorted(metrics.keys()):
            R2 = sym_R2[sym]
            print(f"{sym:>6s} {'ALL':>8s} "
                  f"{sym_bias_mean[sym]:12.6f} {sym_rel_bias_mean[sym]:12.6f} "
                  f"{sym_rmse_mean[sym]:12.6f} {sym_rel_rmse_mean[sym]:12.6f} "
                  f"{sym_R2[sym]:12.6f}")

        # global average
        b = np.mean(list(sym_bias_mean.values()))
        br = np.mean(list(sym_rel_bias_mean.values()))
        r = np.mean(list(sym_rmse_mean.values()))
        rr = np.mean(list(sym_rel_rmse_mean.values()))
        R2 = np.mean(list(sym_R2.values()))

        print('\n# Global averages:')
        print(thead)
        sym = 'ALL'
        print(f"{sym:>6s} {'ALL':>8s} "
              f"{b:12.6f} {br:12.6f} "
              f"{r:12.6f} {rr:12.6f} {R2:12.6f}")

        for symbol in sorted(scatter):
            with open(args.scatterout.format(symbol=symbol), 'w') as fout:
                header = (f"# --confA {args.confA}\n"
                          f"# --confB {args.confB}")
                print(header, file=fout)

                for pair in scatter[symbol]:
                    print(pair[0], pair[1], file=fout)


if __name__ == "__main__":
    main()
