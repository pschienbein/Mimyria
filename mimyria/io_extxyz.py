import io
import re
from collections.abc import Mapping
import numpy as np

import ase.io.extxyz
from ase import Atoms


_original_read_extxyz = ase.io.extxyz.read_extxyz
_original_write_extxyz = ase.io.extxyz.write_extxyz


def _quote_extxyz_string(value):
    """Quote a string for an ExtXYZ comment/header line."""
    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"')
    return f'"{value}"'


def _add_label_to_properties(comment, label_array):
    pattern = r'Properties=(?:"([^"]+)"|(\S+))'
    match = re.search(pattern, comment)

    if match is None:
        # Only valid for plain XYZ with exactly label + x/y/z.
        properties = f"species:S:1:{label_array}:S:1:pos:R:3"

        return (
            f"{comment} Properties={properties}"
            if comment
            else f"Properties={properties}"
        )

    properties = match.group(1) or match.group(2)
    parts = properties.split(":")

    if len(parts) % 3:
        raise ValueError(
            f"Invalid ExtXYZ Properties declaration: {properties!r}"
        )

    new_parts = []
    inserted = False

    for name, dtype, count in zip(
        parts[0::3],
        parts[1::3],
        parts[2::3],
    ):
        new_parts.extend([name, dtype, count])

        if name in ("species", "symbols"):
            new_parts.extend([label_array, "S", "1"])
            inserted = True

    if not inserted:
        raise ValueError(
            "ExtXYZ Properties declaration has no species property"
        )

    replacement = "Properties=" + ":".join(new_parts)

    return comment[:match.start()] + replacement + comment[match.end():]


def _relabel_xyz_stream(fd, atom_relabel, label_array):
    if not isinstance(atom_relabel, Mapping):
        raise TypeError(
            "atom_relabel must be a mapping, for example "
            "{'Co2': 'Co', 'Co3': 'Co'}"
        )

    if not label_array.isidentifier():
        raise ValueError(
            f"label_array must be a valid identifier, got {label_array!r}"
        )

    output = io.StringIO()
    frame_number = 0

    while True:
        line = fd.readline()

        if line == "":
            break

        if not line.strip():
            continue

        frame_number += 1

        try:
            natoms = int(line.strip())
        except ValueError as exc:
            raise ValueError(
                f"Frame {frame_number}: expected an XYZ atom count, "
                f"got {line.rstrip()!r}"
            ) from exc

        comment = fd.readline().rstrip("\r\n")

        # Replace an existing Properties=... entry
        comment = _add_label_to_properties(comment, label_array)

        output.write(f"{natoms}\n")
        output.write(comment + "\n")

        for atom_number in range(1, natoms + 1):
            atom_line = fd.readline()

            if atom_line == "":
                raise EOFError(
                    f"Frame {frame_number}: unexpected EOF at atom "
                    f"{atom_number}/{natoms}"
                )

            fields = atom_line.split()

            original_label = fields[0]
            symbol = atom_relabel.get(original_label, original_label)

            output.write(
                " ".join([symbol, original_label, *fields[1:]]) + "\n"
            )

    output.seek(0)
    return output


def read_extxyz_with_relabel(
    fileobj,
    index=-1,
    *,
    atom_relabel=None,
    label_array="atom_label",
    **kwargs,
):
    if atom_relabel is None:
        atom_relabel = getattr(fileobj, "atom_relabel", None)

    label_array = getattr(
        fileobj,
        "atom_label_array",
        label_array,
    )

    # delegate to original ASE extxyz
    if atom_relabel is None:
        # Completely unchanged ASE behaviour.
        yield from _original_read_extxyz(
            fileobj,
            index=index,
            **kwargs,
        )
        return

    converted = _relabel_xyz_stream(
        fileobj,
        atom_relabel=atom_relabel,
        label_array=label_array,
    )

    try:
        yield from _original_read_extxyz(
            converted,
            index=index,
            **kwargs,
        )
    finally:
        converted.close()


def _write_one_extxyz_with_labels(
    fileobj,
    atoms,
    *,
    label_array,
    **kwargs,
):
    labels = atoms.arrays.get(label_array)

    if labels is None:
        _original_write_extxyz(fileobj, atoms, **kwargs)
        return

    labels = np.asarray(labels)

    if labels.shape != (len(atoms),):
        raise ValueError(
            f"{label_array!r} must have shape ({len(atoms)},), "
            f"got {labels.shape}"
        )

    labels = [str(label) for label in labels]

    for i, label in enumerate(labels):
        if not label or any(char.isspace() for char in label):
            raise ValueError(
                f"Invalid atom label at index {i}: {label!r}"
            )

    # Remove the label array so ASE does not write it as an extra column.
    output_atoms = atoms.copy()
    del output_atoms.arrays[label_array]

    buffer = io.StringIO()

    _original_write_extxyz(
        buffer,
        output_atoms,
        **kwargs,
    )

    buffer.seek(0)

    natoms_line = buffer.readline()
    comment_line = buffer.readline()

    fileobj.write(natoms_line)
    fileobj.write(comment_line)

    # ASE normally writes species as the first atom-line field.
    for label in labels:
        line = buffer.readline()

        if not line:
            raise EOFError("Unexpected end of ASE ExtXYZ output")

        fields = line.split()
        fields[0] = label

        fileobj.write(" ".join(fields) + "\n")

    # Preserve anything remaining, although normally there is nothing.
    fileobj.write(buffer.read())


def write_extxyz_with_labels(
    fileobj,
    images,
    *,
    atom_label_array=None,
    **kwargs,
):
    if atom_label_array is None:
        atom_label_array = getattr(
            fileobj,
            "atom_label_array",
            None,
        )

    if isinstance(images, Atoms):
        image_list = [images]
    else:
        image_list = list(images)

    if not any(atom_label_array in atoms.arrays for atoms in image_list):
        return _original_write_extxyz(
            fileobj,
            image_list,
            **kwargs,
        )

    for atoms in image_list:
        _write_one_extxyz_with_labels(
            fileobj,
            atoms,
            label_array=atom_label_array,
            **kwargs,
        )


# this installes the hook, calling it multiple times is harmless
def install_extxyz_relabel_hook():
    if ase.io.extxyz.read_extxyz is not read_extxyz_with_relabel:
        ase.io.extxyz.read_extxyz = read_extxyz_with_relabel
        ase.io.extxyz.write_extxyz = write_extxyz_with_labels
