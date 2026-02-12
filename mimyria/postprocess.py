
import numpy as np
import sys

from ase.io import iread
from ase.geometry import find_mic

import scipy.constants as const
from scipy.optimize import linear_sum_assignment

from .library import CHARGES

# Angstrom 2 Bohr
g_a2b = 1e-10 / const.physical_constants['Bohr radius'][0]


def calculate_dipole(atoms, _charges=dict()):
    charges = CHARGES.copy()
    charges.update(_charges)

    c = np.array([charges[sym] for sym in atoms.symbols])[:, np.newaxis]
    dipole = np.sum(atoms.positions * c, axis=0)

    # unit is Ae
    return dipole


def calculate_apt_from_field(gradients, field_strength):

    forces = dict()
    for item in gradients:
        if 'forces' in gradients[item].arrays:
            forces[item] = gradients[item].arrays['forces']
        else:
            # interpret the positions to be forces
            # this happens regularly if a CP2k force file is read,
            # which does not contain positions
            forces[item] = gradients[item].positions

    px, py, pz = forces['px'], forces['py'], forces['pz']
    mx, my, mz = forces['mx'], forces['my'], forces['mz']

    # dF/dEx
    dx = (px - mx) / (2.0 * field_strength)
    dy = (py - my) / (2.0 * field_strength)
    dz = (pz - mz) / (2.0 * field_strength)

    # now dx contains [dFx/dEx, dFy/dEx, dFz/dEx]
    # now dy contains [dFx/dEy, dFy/dEy, dFz/dEy]
    # now dz contains [dFx/dEz, dFy/dEz, dFz/dEz]
    P = np.stack((dx, dy, dz), axis=-1)

    return P


def calculate_apt_from_displacement(wanniers, displacement, _charges=dict()):
    Ps = []
    for iAtom, atom in enumerate(wanniers):
        # expecting 6 wannier configurations now
        if len(atom) != 6:
            raise RuntimeError("Expecting 6 wannier configurations per atom")

        print(f'Atom {iAtom} ...', flush=True, end='\r')

        px, mx, py, my, pz, mz = atom

        # ensure that no wanniers are wrapped
        mx = unwrap(mx, px)
        my = unwrap(my, py)
        mz = unwrap(mz, pz)

        # dM/dx
        dx = calculate_dipole(px, _charges) - calculate_dipole(mx, _charges)
        # dM/dy
        dy = calculate_dipole(py, _charges) - calculate_dipole(my, _charges)
        # dM/dz
        dz = calculate_dipole(pz, _charges) - calculate_dipole(mz, _charges)

        # stack the apt
        P = np.stack((dx, dy, dz), axis=-1) / (2 * displacement)

        Ps.append(P)

    return np.array(Ps)


# NOTE when comparing PGT from spatial derivative with
#      PGT from field derivative, users have to pay attention
#      to providing the correct units
#      (see pgt-from-spatial-derivative.py)
def calculate_pgt_from_field(gradients,
                             field_strength,
                             normalized_field=False):
    forces = dict()
    for item in gradients:
        if 'forces' in gradients[item].arrays:
            forces[item] = gradients[item].arrays['forces']
        else:
            # interpret the positions to be forces
            # this happens regularly if a CP2k force file is read,
            # which does not contain positions
            forces[item] = gradients[item].positions

    px, py, pz = forces['px'], forces['py'], forces['pz']
    mx, my, mz = forces['mx'], forces['my'], forces['mz']
    pxpy, mxmy = forces['pxpy'], forces['mxmy']
    pxpz, mxmz = forces['pxpz'], forces['mxmz']
    pypz, mymz = forces['pypz'], forces['mymz']
    o = forces['0']

    hd = field_strength**2

    # d^2F/dEx^2
    dxx = (px + mx - 2 * o) / hd
    # d^2F/dEy^2
    dyy = (py + my - 2 * o) / hd
    # d^2F/dEy^2
    dzz = (pz + mz - 2 * o) / hd

    # if the field vector is normalized by the electronic structure code
    # two different displacements are (silently)
    # applied to calculate the numerical difference
    # and a factor of 2 sneeks in for the off-diagonal elements
    scale = 1
    if normalized_field:
        scale = 0.5

    # d^2F/dEx/dEy
    dxy = scale * (pxpy + mxmy - 2 * o) / hd - 0.5 * (dxx + dyy)
    # d^2F/dEx/dEz
    dxz = scale * (pxpz + mxmz - 2 * o) / hd - 0.5 * (dxx + dzz)
    # d^2F/dEy/dEz
    dyz = scale * (pypz + mymz - 2 * o) / hd - 0.5 * (dyy + dzz)

    # stack them up
    A = np.stack((dxx, dxy, dxz), axis=-1)
    B = np.stack((dxy, dyy, dyz), axis=-1)
    C = np.stack((dxz, dyz, dzz), axis=-1)

    # this is [n,f,i,j], thus not correctly aligned to be multiplied by a velocity
    Q_prime = np.stack((A, B, C), axis=-1)
    # this now contains [n, i, j, f],
    # multiplying by v yields dalpha/dt for each atom
    Q = Q_prime.transpose(0, 2, 3, 1)

    return Q


# NOTE when comparing PGT from spatial derivative with
#      PGT from field derivative, users have to pay attention
#      to providing the correct units
#      (see pgt-from-spatial-derivative.py)
def calculate_pgt_from_displacement(polarizabilities, displacement):
    Qs = []

    for iAtom, atom in enumerate(polarizabilities):
        # expecting 6 polarizabilities per atom
        if len(atom) != 6:
            raise RuntimeError("Expecting 6 polarizabilities per atom")

        alphas = []
        for a in atom:
            alphas.append(a.info['_alpha'].reshape((3, 3)))

        px, mx, py, my, pz, mz = alphas

        # dalpha/dx
        dx = (px - mx) / (2 * displacement)
        # dalpha/dy
        dy = (py - my) / (2 * displacement)
        # dalpha/dz
        dz = (pz - mz) / (2 * displacement)

        Q = np.stack((dx, dy, dz), axis=-1)

        Qs.append(Q)

    return np.array(Qs)


def calculate_alpha_from_field(gradients,
                               field_strength,
                               _charges=dict(),
                               ignore_missing=False,
                               always_unwrap=False):

    if not always_unwrap and \
       not getattr(calculate_alpha_from_field, "_warned", False):
        print('[warn] calculate_alpha_from_field: unwrapping deactivated by default. Unwrapping will only be performed if any component of a dipole difference exceeds 10000. This value was chosen manually for liquid water and might be very wrong for other systems!')
        calculate_alpha_from_field._warned = True

    tensor = []
    for comp in ['x', 'y', 'z']:
        try:
            p = gradients[f'p{comp}']
            m = gradients[f'm{comp}']

        except KeyError:
            if ignore_missing:
                tensor.append(np.zeros(3))
                pass
            else:
                err = f'Requiring fp{comp} and fm{comp} to calculate alpha'
                raise KeyError(err)

        else:
            pM = calculate_dipole(p, _charges)
            mM = calculate_dipole(m, _charges)
            dM = (pM - mM) / (2 * field_strength)

            # NOTE:
            # this is a custom detector if unwrapping is needed
            # MIGHT BE SYSTEM DEPENDENT!
            if always_unwrap or np.any(np.abs(dM) > 10000):
                mM = calculate_dipole(unwrap(m, p), _charges)
                dM = (pM - mM) / (2 * field_strength)

            tensor.append(dM)

    alpha = np.column_stack(tensor)

    return alpha * g_a2b


def calculate_alphas_from_traj(gradient_files, field_strength, _charges=dict()):
    """
    Generator that calculates the polarizability tensors from
    a set of trajectories, defining the px, mx, py, my, pz, and mz
    trajectories
    """

    for structures in zip(*(iread(file) for file in gradient_files.values())):
        gradients = {key: struct for key, struct in zip(gradient_files.keys(), structures)}
        alpha = calculate_alpha_from_field(gradients, field_strength, _charges)

        # take any structure, nuclei positions should be all the same
        # when calculating a gradient with field
        atoms = gradients['px']
        # remove all Wanniers
        del atoms[[atom.symbol == 'X' for atom in atoms]]
        # assign alpha
        atoms.info['alpha'] = alpha

        yield atoms


def unwrap(atoms, atoms_ref, debug=False):
    N = len(atoms)
    if (len(atoms) != len(atoms_ref)):
        raise ValueError('unwrap: Number of atoms not equal')

    if not any(atoms_ref.pbc):
        if not getattr(unwrap, 'has_warned', False):
            print('WARNING: unwrap called without setting a cell; not an issue for gas-phase clusters, but check for periodic systems!', file=sys.stderr)
            unwrap.has_warned = True
        return atoms.copy()

    posA = atoms_ref.positions
    posB = atoms.positions
    cell = atoms_ref.cell

    distances = np.zeros((N, N))
    mic_shifts = []

    for i, pa in enumerate(posA):
        d_mic, dr = find_mic(posB - pa, cell, atoms_ref.pbc)
        distances[i] = np.sum(d_mic**2, axis=1)
        mic_shifts.append(d_mic)

    # find optimal assignment
    row_ind, col_ind = linear_sum_assignment(distances)

    sorted_atoms = atoms[col_ind].copy()

    d_mic_total = np.array([mic_shifts[i][j] for i, j in zip(row_ind, col_ind)])
    sorted_atoms.positions = posA + d_mic_total

    return sorted_atoms
