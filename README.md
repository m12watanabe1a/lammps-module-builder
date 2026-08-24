# LAMMPS Module Builder

This repository provides a Task-based workflow to automate building and installing [LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator)](https://docs.lammps.org/) and generating environment modulefiles.
It simplifies building LAMMPS for multiple versions/configurations and supports loading required toolchain modules from a config file before the build starts.

## Features
- Automated downloading and building of LAMMPS from source.
- Support for multiple LAMMPS versions and configurations.
- Generation of modulefiles for easy environment management.
- Optional module preloading from a shell-friendly module list file.
- No project-level Python `requirements.txt` dependency.
- CMake configure options managed with CMakePresets.

## Supported Platforms
- [x] Linux (WSL on Windows)
- [x] macOS

## Requirements
<details>

- curl
- Python 3.10 or higher
- Environment Modules >=v5 or Lmod >=v8
- [Task](https://taskfile.dev/) v3 or higher

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

2. Define the modules to preload in a config file:
```txt
# config/modules.txt
gcc/13.2.0
openmpi/5.0.3
```

If `config/modules.txt` exists, it is used automatically. Otherwise the default file `config/modules.default.txt` is used.

3. Load the installed LAMMPS module using the module command:
```bash
module use ~/.local/opt/modulefiles
module load lammps/stable_22Jul2025_update5
```

To install under another prefix, pass `PREFIX` to Task. For example:
```bash
task PREFIX=/opt
module use /opt/modulefiles
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
By default, the binaries are installed in `~/.local/opt/lammps/<category>_<date>`, and modulefiles are created in `~/.local/opt/modulefiles/lammps/`. Set `PREFIX` to change the common parent directory, for example `PREFIX=/opt`.

### How is the Python package installed without requirements.txt?
The build script (`scripts/install-python.sh`) uses LAMMPS' own `python/install.py` to generate a wheel, then installs that wheel with `python3 -m pip install --target ...` into:

- `<prefix>/lib/python<major.minor>/site-packages`

This repository does not require a project-level `requirements.txt` because there are no extra Python package dependencies for the build workflow itself.

### Which CMake preset template is used?
During `cmake_configure`, the Task workflow selects preset files in this order:

1. `config/CMakePresets.json` (if present)
2. `config/CMakePresets.default.json` (fallback)

The selected file is copied to `cmake/CMakePresets.json` in the LAMMPS source tree.
