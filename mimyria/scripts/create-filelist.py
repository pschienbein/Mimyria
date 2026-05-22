#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

PROPERTY_FIELDS = {
    "elem": "species:S:1",
    "species": "species:S:1",
    "pos": "pos:R:3",
    "vel": "vel:R:3",
    "force": "force:R:3",
    "forces": "force:R:3",
    "apt": "apt:R:9",
    "pgt": "pgt:R:27",
}


def make_properties(contains: str) -> str:
    fields = []

    for item in contains.split(","):
        item = item.strip().lower()
        if not item:
            continue

        if item not in PROPERTY_FIELDS:
            known = ", ".join(sorted(PROPERTY_FIELDS))
            raise ValueError(f"Unknown property field '{item}'. Known fields: {known}")

        fields.append(PROPERTY_FIELDS[item])

    if not fields:
        raise ValueError("Empty --contains argument")

    return "Properties=" + ":".join(fields)


def add_properties(path: str, contains: str) -> str:
    if contains is None:
        return path
    return f"{path} ({make_properties(contains)})"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a trajectory file list from ordered trajectory file groups.\n\n"
            "Each --file argument adds one group of trajectory files. "
            "All groups must contain the same number of files. "
            "For each trajectory index, one output line is written by taking "
            "one file from each group in the order in which the groups were given.\n\n"
            "Use --contains after a --file group to append synthesized extxyz "
            "Properties=... metadata to that group. If --contains is omitted, "
            "the file is written without additional metadata.\n\n"
            "Example:\n"
            "  --file run*/apt.xyz.zst --contains elem,pos,apt \\\n"
            "  --file run*/vel.xyz     --contains elem,vel\n\n"
            "produces entries like:\n"
            "  run001/apt.xyz.zst (Properties=species:S:1:pos:R:3:apt:R:9) , "
            "run001/vel.xyz (Properties=species:S:1:vel:R:3) "
            "that are then treated as a single trajectory\n\n"
            "Recognized property fields:\n"
            "  elem/species -> species:S:1\n"
            "  pos          -> pos:R:3\n"
            "  vel          -> vel:R:3\n"
            "  force        -> force:R:3\n"
            "  apt          -> apt:R:9\n"
            "  pgt          -> pgt:R:27\n"
            "  from-file    -> read metadata directly from the trajectory\n"
            "  -            -> alias for from-file\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--file",
        action="append",
        nargs="+",
        required=True,
        help=(
            "One ordered group of trajectory files. "
            "May be given multiple times."
        ),
    )

    parser.add_argument(
        "--contains",
        action="append",
        required=True,
        help=(
            "Comma-separated fields for the corresponding --file group. "
            "Must be given once per --file group. Use 'none' for no added metadata."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        default="filelist",
        help="Output file name [default: filelist]",
    )

    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Store absolute instead of relative paths [default: False]"
    )

    args = parser.parse_args(argv)

    file_groups = args.file
    contains_groups = args.contains

    if len(contains_groups) != len(file_groups):
        raise SystemExit(
            "Error: provide exactly one --contains argument per --file argument.\n"
            f"Got {len(file_groups)} --file groups and "
            f"{len(contains_groups)} --contains groups.\n\n"
            "Example:\n"
            "  --file apt.xyz.zst --contains from-file "
            "--file vel.xyz --contains elem,vel"
        )

    contains_groups = [
        None if c.lower() in {"none", "no", "false", "-", "from-file"} else c
        for c in contains_groups
    ]

    while len(contains_groups) < len(file_groups):
        contains_groups.append(None)

    lengths = [len(group) for group in file_groups]

    if len(set(lengths)) != 1:
        raise SystemExit(
            "Error: all --file groups must contain the same number of files. "
            f"Got lengths: {lengths}"
        )

    ntraj = lengths[0]

    # PRINT SOME INFORMATION
    print(
        f"[INFO] Number of trajectory groups : {len(file_groups)}",
        file=sys.stderr,
    )

    print(
        f"[INFO] Number of trajectories      : {ntraj}",
        file=sys.stderr,
    )

    print(
        f"[INFO] Output file                 : {args.output}",
        file=sys.stderr,
    )

    print("[INFO] First trajectory preview:", file=sys.stderr)

    preview_entries = []

    for idx, (files, contains) in enumerate(zip(file_groups, contains_groups)):
        entry = add_properties(files[0], contains)

        preview_entries.append(entry)

        print(
            f"  Group {idx + 1}:",
            file=sys.stderr,
        )
        print(
            f"    first file: {files[0]}",
            file=sys.stderr,
        )

        if contains is None:
            print(
                f"    properties: <Retrieved from trajectory file>",
                file=sys.stderr,
            )
        else:
            print(
                f"    properties: {make_properties(contains)}",
                file=sys.stderr,
            )

    print(
        "\n[INFO] First output line:",
        file=sys.stderr,
    )

    print(
        "  " + " , ".join(preview_entries),
        file=sys.stderr,
    )

    print(file=sys.stderr)

    # PROCEED WITH WRITING THE FILELIST
    with open(args.output, "w") as handle:
        for i in range(ntraj):
            entries = []

            for files, contains in zip(file_groups, contains_groups):
                if contains == "none":
                    contains = None

                if args.absolute_paths:
                    entries.append(add_properties(str(Path(files[i]).resolve()), contains))
                else:
                    entries.append(add_properties(files[i], contains))

            handle.write(" , ".join(entries) + "\n")


if __name__ == "__main__":
    main()

