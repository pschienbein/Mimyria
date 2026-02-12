from ase import Atoms
from ase.utils.plugins import ExternalIOFormat

from importlib.metadata import EntryPoint

import numpy as np

import io
from pathlib import Path

import importlib
zstd_spec = importlib.util.find_spec('zstandard')
if zstd_spec:
    import zstandard as zstd
else:
    zstd = None


# Wrapper to open a compressed file
def open_wrapper(path, mode='rt'):
    suffixes = [s.lower() for s in Path(path).suffixes]
    suffix = suffixes[-1]

    if suffix == ".xz":
        if not importlib.util.find_spec('lzma'):
            print('[ERR] recognized xz extension for {fn}, but lzma module is not installed!')
            raise ModuleNotFoundError('lzma module missing')
        import lzma

        # default is inversed...
        if 't' not in mode and 'b' not in mode:
            mode += 't'

        if 'b' in mode:
            fh = lzma.open(path, mode)
        else:
            fh = lzma.open(path, mode, encoding='utf-8')

        del suffixes[-1]

    elif suffix in (".zst", ".zstd"):
        if not importlib.util.find_spec('pyzstd'):
            print('[ERR] recognized zstd extension for {fn}, but zstandard module is not installed!')
            raise ModuleNotFoundError('zstandard module missing')
        import pyzstd as zstd

        if 'r' in mode:
            rdr = zstd.ZstdFile(path, 'r')
            if 'b' in mode:
                fh = rdr
            else:
                fh = io.TextIOWrapper(rdr, encoding='utf-8')

        else:
            wtr = zstd.ZstdFile(path, mode)
            if 'b' in mode:
                fh = wtr
            else:
                fh = io.TextIOWrapper(wtr, encoding='utf-8')

        del suffixes[-1]

    else:
        # just a 'default' file
        fh = open(path, mode)

    # assign the ase_format to the opened stream
    ase_format = suffixes[-1].replace('.', '')
    if ase_format == 'xyz':
        ase_format = 'extxyz'
    setattr(fh, 'ase_format', ase_format)

    return fh


def apt_flatten(apt):
    return apt.reshape(len(apt), 9)


def apt_reshape(apt):
    if apt.ndim == 2 and apt.shape[1] == 9:
        return apt.reshape(-1, 3, 3)
    else:
        return apt


def pgt_flatten(pgt):
    return pgt.reshape(len(pgt), 27)


def pgt_reshape(pgt):
    if pgt.ndim == 2 and pgt.shape[1] == 27:
        return pgt.reshape(-1, 3, 3, 3)
    else:
        return pgt


def atoms_arrays_flatten(atoms):
    if 'apt' in atoms.arrays:
        atoms.arrays['apt'] = apt_flatten(atoms.arrays['apt'])
    if 'pgt' in atoms.arrays:
        atoms.arrays['pgt'] = pgt_flatten(atoms.arrays['pgt'])


def atoms_arrays_reshape(atoms):
    if 'apt' in atoms.arrays:
        atoms.arrays['apt'] = apt_reshape(atoms.arrays['apt'])
    if 'pgt' in atoms.arrays:
        atoms.arrays['pgt'] = pgt_reshape(atoms.arrays['pgt'])


PolarIOFormat = ExternalIOFormat(
        desc='polar file format',
        code='+F',
        module='mimyria.io',
        ext='polar'
        )


#ep = EntryPoint(
#        name='polar',
#        value='mimyria.io:PolarIOFormat',
#        group='ase.ioformats'
#        )


def read_polar(file, index=-1, **kwargs):
    """
    Read a cp2k "polar" file which contains the polarizability tensor
    along the trajectories without nuclei coordinates
    Thus returns an empty Atoms object containing the
    polarizability tensor only
    """
    for line in file:
        if line.find('#') >= 0:
            continue

        aLine = line.split()
        alpha = np.array(aLine[2:]).astype(float)

        atoms = Atoms()
        atoms.info['_i'] = int(aLine[0])
        atoms.info['_time'] = float(aLine[1])
        atoms.info['_alpha'] = alpha

        yield atoms


def write_polar(fd, frames, **kwargs):
    if not isinstance(frames, (list, tuple)):
        frames = [frames]

    # header?
    if fd.tell() == 0:
        fd.write('#   Step   Time [fs]           xx [a.u.]           xy [a.u.]           xz [a.u.]           yx [a.u.]           yy [a.u.]           yz [a.u.]           zx [a.u.]           zy [a.u.]           zz [a.u.]\n')

    for i, atoms in enumerate(frames):
        alpha = atoms.info['_alpha']

        fd.write(f'{i} 0 ')
        for f in alpha:
            fd.write(f'{f} ')
        fd.write('\n')
