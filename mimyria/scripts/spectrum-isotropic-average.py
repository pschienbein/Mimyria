#!/bin/env python3

import argparse
import re

import numpy as np


def load_data(input_file, expected):
    # create an array [spectrum_id (CCA)][geometry][time]
    data = []
    time_series = None

    # is it a single file?
    if '{}' not in input_file:
        # load all from this file, first get the comment
        with open(input_file) as fin:
            for line in fin:
                if line.lstrip().startswith('#'):
                    if 'Time' in line:
                        col_info_line = line.rstrip('\n')
                else:
                    break

        aCols = col_info_line.split('|')
        col2idx = {}
        for idx, name in enumerate(aCols):
            m = re.match(r"\s*(AC|CC)\((.+)\)\s*", name)
            if m:
                n = m.group(2).replace(',', '')
                col2idx[n] = idx

        # test if all expected columns are present
        for exp in expected:
            if exp not in col2idx:
                print(f'ERROR: Expecting correlation function in direction {exp}, but could not find it in the given input file')
                print('       Maybe the different geometries are written in different files?')
                exit(1)

        # now load the data
        this_data = np.loadtxt(input_file)

        time_series = this_data[:, 0]

        data.append({})
        for geo in expected:
            data[0][geo] = this_data[:, col2idx[geo]]

    # It is a template
    else:
        num_spectra = 0
        for geo in expected:
            # try to open the file
            try:
                fn = input_file.format(geo)
                this_data = np.loadtxt(fn)

                # extract number of spectra and consistency check
                this_num_spectra = this_data.shape[1] - 1
                if this_num_spectra != num_spectra and num_spectra != 0:
                    print(f'ERROR: Number of spectra in {fn} is inconsistent to the previously loaded files')
                    exit(1)
                num_spectra = this_num_spectra

                # store time series
                if time_series is None:
                    time_series = this_data[:, 0]
                else:
                    # check if it's consistency
                    if not np.all(time_series == this_data[:, 0]):
                        print(f'ERROR: Times stored in {fn} are inconsistent to the previously loaded files')
                        exit(1)

                # array not initialized?
                if len(data) == 0:
                    for iSpec in range(num_spectra):
                        data.append({})

                # append this data to the array:
                for iSpec in range(num_spectra):
                    data[iSpec][geo] = this_data[:, iSpec + 1]

            except FileNotFoundError:
                print(f'ERROR: Expecting correlation function in direction {exp}, but the file {fn} could not be opened')
                exit(1)

    return time_series, data


def main(argv=None):
    parser = argparse.ArgumentParser(description='''Takes the output of mimyria and computes the isotropic average for IR or Raman spectra.
            NOTE: Averaging is done at the level of the correlation functions, the spectra can subsequently be obtained by corr2spec.py''')
    parser.add_argument('--spectrum_kind', type=str, choices=['ir', 'raman'], required=True, help='Average IR or Raman spectrum')
    parser.add_argument('--inp', type=str, required=True,
                        help='Can be a placeholder if geometries are stored in different files (e.g. pgt2cf-{}.cf)')
    parser.add_argument('--out', type=str, required=True,
                        help='Can be a placeholder if geometries should be stored in different files (e.g. pgt2cf-{}.cf)')

    args = parser.parse_args(argv)

    if args.spectrum_kind == 'ir':
        expecting = ['x', 'y', 'z']

        # read in correlation functions
        times, data = load_data(args.inp, expecting)
        print(times.shape)
        print(len(data))

        result = [times]

        # loop over all spectra
        for iSpec in range(len(data)):
            # do the averaging:
            iso = data[iSpec]['x'] + data[iSpec]['y'] + data[iSpec]['z']
            iso /= 3.0

            result.append(iso)

        arr = np.array(result).T
        np.savetxt(args.out, arr, fmt='%.6e')

    elif args.spectrum_kind == 'raman':
        expecting = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'xxyy', 'yyzz', 'xxzz']

        times, data = load_data(args.inp, expecting)

        result = {'iso': [times], 'aniso': [times], 'vv': [times], 'vh': [times]}

        # loop over all spectra
        for iSpec in range(len(data)):
            # According to Berne...
            iso = (data[iSpec]['xx'] + data[iSpec]['yy'] + data[iSpec]['zz'] +
                   2.0 * (data[iSpec]['xxyy'] + data[iSpec]['xxzz'] + data[iSpec]['yyzz'])) / 9.0

            vh = (data[iSpec]['xx'] + data[iSpec]['yy'] + data[iSpec]['zz'] +
                  2.0 * (data[iSpec]['xy'] + data[iSpec]['xz'] + data[iSpec]['yz']) -
                  (data[iSpec]['xx'] + data[iSpec]['yy'] + data[iSpec]['zz'] +
                   2.0 * (data[iSpec]['xxyy'] + data[iSpec]['xxzz'] + data[iSpec]['yyzz'])) / 3.0) / 10.0

            vv = iso + 4.0 * vh / 3.0

            aniso = vh / 15.0

            result['iso'].append(iso)
            result['aniso'].append(aniso)
            result['vv'].append(vv)
            result['vh'].append(vh)
     
        if "{}" in args.out:
            for key in result:
                arr = np.array(result[key]).T
                np.savetxt(args.out.format(key), arr, fmt='%.6e')

        else:
            # ONLY a single spectrum
            if len(data) == 1:
                iso = result['iso'][1]
                aniso = result['aniso'][1]
                vv = result['vv'][1]
                vh = result['vh'][1]

                arr = np.column_stack((times, iso, aniso, vv, vh))
                header = '# Time | ISO | ANISO | VV | VH'
                np.savetxt(args.out, arr, fmt='%.6e', header=header)

            else:
                print('# isotropic spectra should be written in different files '
                      '(more than one spectrum is provided), '
                      'please provide a placeholder {} in the output file string')
                exit(1)


if __name__ == "__main__":
    main()
