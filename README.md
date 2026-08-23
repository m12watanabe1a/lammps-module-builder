# LAMMPS Module Builder

This repository provides a Task-based workflow to automate building and installing [LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator)](https://docs.lammps.org/) and generating environment modulefiles.
It simplifies building LAMMPS for multiple versions/configurations and supports optional module preloading before the build starts.

## Features
- Automated downloading and building of LAMMPS from source.
- Support for multiple LAMMPS versions and configurations.
- Generation of modulefiles for easy environment management.
- Optional preloading of dependency modules (for compiler/MPI/toolchain environments).
- No project-level Python `requirements.txt` dependency.
- CMake configure options managed with CMakePresets.

## Supported Platforms
- [x] Linux (WSL on Windows)
- [x] macOS

## Requirements
<details>

- curl
- Python 3.10 or higher
- lmod v8.4 or higher
- gpatch (Only for macOS)

Also you need to have development tools and libraries installed for building LAMMPS, such as:
- C++ compiler (e.g., GCC, Clang)
- CMake
- MPI library (e.g., OpenMPI, MPICH)
- FFT library (e.g., FFTW)
- Other dependencies as required by specific LAMMPS packages.
Please refer to the [LAMMPS building guide](https://docs.lammps.org/Build_cmake.html) for detailed information on required dependencies.
</details>

## Install LAMMPS Modules
1. Run the default task (`all`) to build/install configured targets and generate modulefiles:
```bash
task
```

The build steps are defined in `Taskfile.yml`. You can inspect the available tasks with:
```bash
task --list
```

2. Optionally preload environment modules before running the build:
```bash
task -- gcc/13.2.0 openmpi/5.0.3
```

The arguments after `--` are passed to `module load ...` before invoking the build Taskfile.

3. Load the installed LAMMPS module using the module command:
```bash
module use ~/.local/opt/modulefiles
module load lammps/stable_22Jul2025_update5
```

## FAQ
(Although I haven’t actually received any questions yet.)

### How do I build a specific version/category?
Use Task variables when invoking tasks:
```bash
task target CATEGORY=stable DATE=22Jul2025_update5
```

The default values are defined in `Taskfile.yml` and `Taskfile.lammps.yml`.

### Where are the built LAMMPS binaries and modulefiles located?
By default, the binaries are installed in `~/.local/opt/lammps/<category>_<date>`, and modulefiles are created in `~/.local/opt/modulefiles/lammps/`.

### How is the Python package installed without requirements.txt?
The build script (`scripts/install-python.sh`) uses LAMMPS' own `python/install.py` to generate a wheel, then installs that wheel with `python3 -m pip install --target ...` into:

- `<prefix>/lib/python<major.minor>/site-packages`

This repository does not require a project-level `requirements.txt` because there are no extra Python package dependencies for the build workflow itself.

### Which CMake preset template is used by default?
The Task workflow uses `config/CMakePresets.default.json` as the template and copies it to `cmake/CMakePresets.json` in the LAMMPS source tree during `cmake_configure`.
