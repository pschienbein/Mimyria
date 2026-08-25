
# Changelog

## [0.9.4]

### Added

- Added *--enforce_force_sum_rule* option for *pgt-from-efield-derivative* script.
- *pgt-from-efield-derivative* script can now also process compressed trajectory files.
- Rewrote *spectrum-isotropic-averages* script to preserve comments.
- Updated input file templates for newest version of CP2k (&REFTRAJ EVAL ENERGY_FORCES)
- Added support for custom atom symbol labels, such as Co2, Fe3, ...

### Fixed

- Now correctly shipping the cp2k templates file for non-editable python installations.
- Fixed printing the correct git information string when HPC nodes don't have git installed.

## [0.9.2]

### Added

- Added **mimyria-py create-filelist** to conveniently create filelists as input for **mimyria**.

### Fixed

- Fixed extremely slow on-the-fly decompression of `.zst` and `.xz` files with ASE readers caused by frequent seek operations, which could dominate prediction runtime.
- Printing the git commit version after installation with pip
- Parsing charges in **mimyria-py apt_from_spatial_derivative**
- Many cosmetics
