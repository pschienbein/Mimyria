#!/bin/env python3

import argparse
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare two files containing APTs and provide the rmse')
    parser.add_argument('--specA', type=str, required=True)
    parser.add_argument('--specB', type=str, required=True)
    parser.add_argument('--logscale', action='store_true', default=False, help='Apply the log scale to the spectra before comparing. NOTE: Cross correlations can be negative, where this comparison is not meaningful!')
    parser.add_argument('--maxfreq', default=None, type=float, help='Maximum frequency up to which the two spectra are to compared')

    args = parser.parse_args(argv)
    #####################################################

    specA = np.loadtxt(args.specA)
    specB = np.loadtxt(args.specB)

    min_rows = min(specA.shape[0], specB.shape[0])
    specA = specA[:min_rows, :]
    specB = specB[:min_rows, :]

    specA = specA.transpose()
    specB = specB.transpose()

    if len(specA) != len(specB):
        print('Number of columns in the files do not match')
        exit(1)

    if args.logscale:
        print('# Performing comparison on the logarithmic scale. For all negative values the ABSOLUTE is taken which might not be sensible!')

    if args.maxfreq is not None:
        cutoff_idx = np.searchsorted(specA[0], args.maxfreq)
        specA = specA[:, :cutoff_idx]
        specB = specB[:, :cutoff_idx]

    scores = []
    indices = []
    for col in range(1, len(specA)):
        if args.logscale:
            if np.all(np.abs(specA[col]) > 1e-8) and np.all(np.abs(specB[col]) > 1e-8):
                specA[col] = np.log(np.abs(specA[col]))
                specB[col] = np.log(np.abs(specB[col]))
            else:
                specA[col] = np.zeros(len(specA[col]))
                specB[col] = np.zeros(len(specA[col]))

        diff = np.abs(specA[col] - specB[col])
        s = np.abs(specA[col] + specB[col])
        num = np.sum(diff)
        den = np.sum(s)

        # there can be empty columns
        if den > 1e-10:
            score = num / den
            print(f'Score Column: {col+1}: {score:.5f} -- {(1 - score) * 100:.1f}%')
            scores.append(score)
            indices.append(col)
        else:
            print(f'Score Column: {col+1}: NULL')

    avg_score = np.mean(scores)
    score_spread = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    min_col = indices[np.argmin(scores)] + 1
    max_col = indices[np.argmax(scores)] + 1
    print(f'Min | Max | Average | Spread Score: '
          f'{min_score:.5f} {max_score:.5f} '
          f'{avg_score:.5f} {score_spread:.5f}')

    print(f'Percentage                        : '
          f'{(1 - min_score) * 100:.5f} {(1 - max_score) * 100:.5f} '
          f'{(1 - avg_score) * 100:.5f} {score_spread * 100:.5f}')

    print(f'Min Score Column: {min_col}')
    print(f'Max Score Column: {max_col}')


if __name__ == '__main__':
    main()
