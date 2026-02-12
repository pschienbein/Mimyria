#!/bin/env python3

import argparse
import numpy as np
import scipy.constants as phys_const
import sys

from scipy.fft import dct


def main(argv=None):
    parser = argparse.ArgumentParser(description='Takes correlation functions and computes the IR spectrum')
    parser.add_argument('--corr_in', type=str, required=True)
    parser.add_argument('--spec_out', type=str, required=True)

    parser.add_argument('--norm',
                        type=str,
                        choices=['ir', 'raman', 'integral', 'none'],
                        default='none',
                        help='The employed normalization. ir: Returns the proper IR absorption coefficient; integral: Normalizes the total spectrum by its integral')

    parser.add_argument('--timestep', type=float, default=1.0, help='Timestep of the correlation function in fs (default: 1fs)')
    parser.add_argument('--window_length', type=int, default=1000, help='Hann-window taper length in lags (default: 1000). Larger taper lengths retain more of the signal’s fine structure; 0 evaluates to the full number of lags, i.e. minimal tapering.')
    parser.add_argument('--temperature', type=float, default=300, help='Simulation temperature, only used if --norm=ir (default: 300K)')
    parser.add_argument('--volume', type=float, default=1.0, help='Volume of the simulation box, only used if --norm=ir (default: 1 Angstrom^3); note: the spectrum is normalized by the given volume, 1 deactivates this normalization')

    args = parser.parse_args(argv)
    #####################################################


    # read the file
    data = np.loadtxt(args.corr_in)

    # layout of the CF file:
    # 1: Index
    # N: Correlation functions
    data = data.transpose()
    Ncf = len(data) - 1

    print(f'Found {Ncf} correlation functions, converting each of them to a spectrum...')

    # compute some properties
    C0 = phys_const.speed_of_light
    # Nyquist frequency (in cm-1):
    nu_c = 1.0 / (args.timestep * 1e-15) / C0 / 200.0


    # Compute the frequencies of the DCT
    N = len(data[0])
    # conversion to cm^-1
    c_cm_fs = 1e-15 * C0 * 100.0
    freqs = np.arange(N) / (2 * (N - 1) * args.timestep) / c_cm_fs
    out_data = [freqs]

    # report input values
    header = f'# Timestep: {args.timestep} fs\n'
    header += f'# Temperature: {args.temperature} K\n'
    header += f'# Hann Window Length: {args.window_length}\n'
    header += f'# Normalization: {args.norm}\n'
    header += f'# Nyquist-Frequency: {nu_c} cm^{-1}\n'
    if args.norm == 'ir':
        header += '# IR absorption coefficient (alpha * n) printed in units of cm^{-1}\n'
        header += '#   assuming APTs were given in e\n'
        header += '#   and velocities were given in Angstrom fs^{-1}\n'
    elif args.norm == 'raman':
        header += '# Raman lineshape printed in units of e^4 a0^4 Eh^{-2} fs^{-2} cm\n'
        header += '#   assuming PGTs were given in e^2 a0^2 Eh^{-1} Angstrom^{-1}\n'
        header += '#   and velocities were given in Angstrom fs^{-1}\n'
    print(header, end='')

    # valid for norm none
    aconv = 1.0
    if args.norm == 'ir':
        # conversion of the absorption coefficient
        # Note:
        # 1) Unit of correlation function is e^2 A^2 / fs^2
        # 2) 1e23: conversion from A to cm and 1/fs to 1/s
        num = args.timestep * np.pi * 1e23 * phys_const.e**2
        den = 6 * args.volume * C0 * phys_const.epsilon_0 * phys_const.Boltzmann * args.temperature
        aconv = num / den

        # INFO
        print('# INFO: Normalization ir:\n'
              '#       To function as intended, the timestep, volume and temperature need to be provided\n'
              '#       Also it is assumed that APTs were given in elementary charge and velocities in A/fs for calculating the correlation functions')

    elif args.norm == 'raman':
        # conversion of the Raman TCF:
        # Note:
        # 1) Unit of the TCF: e^4 * a_0^4 / Eh^2 / fs^2
        # 2) 1e-13: conversion of C0 from m/s to cm/fs
        aconv = args.timestep * c_cm_fs  # unit: cm

        # INFO
        print('# INFO: Normalization raman:\n'
              '#       To function as intended, the timestep needs to be provided\n'
              '#       Also it is assumed that PGTs were given in e^2 a0^2 / Eh / A (convention)  and velocities in A/fs for calculating the correlation functions')

    for iCF, cf in enumerate(data[1:len(data)]):
        print(f'Processing Correlation Function {iCF + 1}', file=sys.stderr)
        Nlags = len(cf)
        if args.window_length == 0:
            wlen = Nlags
        else:
            wlen = min(Nlags, args.window_length)

        for i in range(len(cf)):
            if i > wlen:
                cf[i] = 0.0
            else:
                # apply Hann window
                cf[i] *= (1.0 - np.sin(0.5 * (i-1.0) * np.pi / wlen)**2)

        spec = dct(cf, type=1, norm=None)

        # normalize by integral
        if args.norm == 'integral':
            aconv = 1.0 / np.sum(np.abs(spec))

        # apply normalization:
        spec *= aconv

        out_data.append(spec)

    fmt = tuple('%.6e' for _ in range(len(out_data)))
    out_data = np.array(out_data).transpose()

    header += '# 1: Frequency in cm^{-1}\n'
    header += f'# 2 - N: Spectra for all columns contained in {args.corr_in}\n'

    np.savetxt(args.spec_out, out_data, header=header, fmt=fmt, comments='')


if __name__ == "__main__":
    main()
