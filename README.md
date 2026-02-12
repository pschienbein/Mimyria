# Mimyria

Machine Learning Accelerated Theoretical Condensed-Phase Spectroscopy Made Simple

## Introduction

**Mimyria** is a modular toolkit for computing theoretical IR and Raman spectra from molecular dynamics (MD) simulations using machine-learning acceleration. It is **independent of the underlying interaction model** and can be used on top of **any existing MD workflow**, including _ab initio_ MD or simulations driven by machine learning potentials.  
Mimyria is intended as a **post-processing layer** and can be applied after the MD simulations and interaction model are finalized, allowing IR and Raman spectra to be generated only if and when they are needed, **without requiring any changes to the interaction potential**.

Starting from existing MD trajectories—generated with an arbitrary ML or _ab initio_ potential—Mimyria automatically performs the required electronic structure calculations, trains ML models on the fly, and predicts IR and Raman spectra. For convenience, an _autotrain_ script is provided that takes existing trajectories as input and produces the final spectra with minimal user intervention.

For a detailed description of capabilities, methodology, and best practices, please see the accompanying paper https://doi.org/10.48550/arXiv.2602.06760.

## Installation

**Mimyria** consists of two fully independent layers: a Python-based machine learning layer and a C++ spectrum post-processing layer.

### Python / Machine Learning Layer

Mimyria depends on **e3nn**, which in turn depends on **PyTorch**. The package has been tested with **Python 3.10**.

If you already have a working PyTorch installation (CPU or CUDA), clone the git repository and install Mimyria with:

`pip install .`

**Important:** If PyTorch is not installed, `pip` will automatically install a compatible version to satisfy `e3nn`, which may result in a **CPU-only build**.  
For GPU acceleration, install a CUDA-enabled PyTorch build first, or use one of the provided requirement files:

`pip install -r requirements-cpu.txt pip install -r requirements-cu118.txt pip install -r requirements-cu126.txt pip install -r requirements-cu128.txt`

The requirement files install a matching PyTorch build and then install Mimyria in editable mode.

After installation, the python layer is callable via `mimyria-py`

### C++ / Spectrum Post-Processing Layer

The spectrum post-processing layer converts learned electronic structure information into IR and Raman spectra. It is implemented in C++ and requires **Eigen**, **FFTW3**, and a **C++23–compatible compiler** (GCC 11 or newer). These dependencies are expected to be available on the system.

Compile the C++ layer by running:

`make`

in the root repository directory.

Both layers are **fully independent** and can be installed and used separately; users may use only the machine learning layer or only the spectrum post-processing layer, depending on their workflow.

An **example environment modulefile** is provided in the repository, which sets the required `PATH` and `LD_LIBRARY_PATH` entries to ensure that the compiled executables and libraries are found correctly. This is particularly useful on HPC systems using environment modules.

After installation, the C++ layer of mimyria is callable via the command `mimyria`
## Units

While units are generally irrelevant for machine learning potentials, we have to make a choice to calculate absolute intensities and consistent Raman scattering lineshape functions.
All scripts that calculate and postprocess APTs and/or PGTs use the following convention:

| Symbol                                    | Unit                                      | Output of                                                                |
| ----------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------ |
| velocity                                  | $Å$ $\text{fs}^{-1}$                      | Your favourite MD code                                                   |
| force                                     | $E_h$ $a_0^{-1}$                          | Your favourite electronic structure code                                 |
| electric field                            | $E_h$ $a_0^{-1}$$e^{-1}$                  | Your favourite electronic structure code                                 |
| APT                                       | $e$                                       | *mimyria-py*:<br>apt-from-efield-derivative, apt-from-spatial-derivative |
| $\lt \dot{M}(0) \dot{M}(t) \gt$           | $e^2$ $Å^2$ $\text{fs}^{-2}$              | *mimyria*:<br>apt2cf                                                     |
| electric<br>polarizability                | $e^2$ $a_0^2$ $E_h^{-1}$                  |                                                                          |
| PGT                                       | $e^2$ $a_0^2$ $E_h^{-1}$ $Å^{-1}$         | *mimyria-py*:<br>pgt-from-efield-derivative, pgt-from-spatial-derivative |
| $\lt \dot{\alpha}(0) \dot{\alpha}(t) \gt$ | $e^4$ $a_0^4$ $E_h^{-2}$ $\text{fs}^{-2}$ | *mimyria*: <br>pgt2cf                                                    |

Mostly atomic units are employed, with the exception of positions and velocities, that are usually expressed in units of Angstrom and femtoseconds in MD codes and thus require an adjustment.
Importantly, all low-level scripts related to training (*mimyria-py train*), prediction (*mimyria-py predict*), and comparing APTs and/or PGTs (*mimyria-py compare*) do not assume any units, such that the ML part can readily be embedded into other post-processing workflows that work with a different unit convention.
