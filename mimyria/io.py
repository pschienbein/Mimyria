from ase import Atoms
from ase.utils.plugins import ExternalIOFormat

import numpy as np

import io
import tempfile
import atexit
import errno
import os
from pathlib import Path

from mimyria.io_extxyz import install_extxyz_relabel_hook

import importlib
zstd_spec = importlib.util.find_spec('zstandard')
if zstd_spec:
    import zstandard as zstd
else:
    zstd = None


_TEMP_FILES = set()


def _register_tmp(path):
    _TEMP_FILES.add(path)


def _unregister_tmp(path):
    _TEMP_FILES.discard(path)


@atexit.register
def _cleanup_tmp_files():
    for path in list(_TEMP_FILES):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            pass


# decompress to tmp file
def _decompress_to_temp(path, opener):
    last_exc = None

    for tmp_dir in (None, os.getcwd()):
        tmp_path = None

        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir)
            tmp_path = tmp.name
            tmp.close()

            _register_tmp(tmp_path)

            with opener(path, 'rb') as src, open(tmp_path, 'wb') as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)

            return tmp_path

        except OSError as exc:
            last_exc = exc
            is_no_space = exc.errno == errno.ENOSPC

            # fallback to working dir
            if tmp_dir is None and is_no_space:
                continue

            raise

    raise last_exc


# Wrapper to open a compressed file
def open_wrapper(path,
                 mode='rt',
                 atom_relabel=None,
                 atom_label_array="atom_label"):
    suffixes = [s.lower() for s in Path(path).suffixes]
    suffix = suffixes[-1]

    tmp_path = None

    install_extxyz_relabel_hook()

    if suffix == ".xz":
        if not importlib.util.find_spec('lzma'):
            print('[ERR] recognized xz extension for {fn}, but lzma module is not installed!')
            raise ModuleNotFoundError('lzma module missing')
        import lzma

        tmp_path = _decompress_to_temp(path, lzma.open)
        fh = open(tmp_path, mode)

        del suffixes[-1]

    elif suffix in (".zst", ".zstd"):
        if not importlib.util.find_spec('pyzstd'):
            print('[ERR] recognized zstd extension for {fn}, but zstandard module is not installed!')
            raise ModuleNotFoundError('zstandard module missing')
        import pyzstd as zstd

        if 'r' in mode:
            tmp_path = _decompress_to_temp(path, zstd.ZstdFile)
            fh = open(tmp_path, mode)

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

    # cleanup
    if tmp_path is not None:
        orig_close = fh.close

        def _close_and_cleanup():
            try:
                orig_close()
            finally:
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass

        fh.close = _close_and_cleanup

    # assign the ase_format to the opened stream
    ase_format = suffixes[-1].replace('.', '')
    if ase_format == 'xyz':
        ase_format = 'extxyz'
    setattr(fh, 'ase_format', ase_format)

    if atom_relabel is not None:
        setattr(fh, 'atom_relabel', dict(atom_relabel))
    setattr(fh, 'atom_label_array', atom_label_array)

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

