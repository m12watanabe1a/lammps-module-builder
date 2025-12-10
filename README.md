# LAMMPS Module Builder

This repository provides a Python-based tool to automate the building and installation of [LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator)](https://docs.lammps.org/) modules.
It simplifies the process of compiling LAMMPS with various version, configurations and managing module files for easy loading in different environments.

## Features
- Automated downloading and building of LAMMPS from source.
- Support for multiple LAMMPS versions and configurations.
- Generation of module files for easy environment management.
- Customizable build recipes and target configurations.

## Supported Platforms
- [x] Linux (WSL on Windows)
- [x] macOS

## Requirements
<details>

- Python 3.8 or higher
  - Pyyaml
  - Jinja2
- lmod v8.4 or higher

Also you need to have development tools and libraries installed for building LAMMPS, such as:
- C++ compiler (e.g., GCC, Clang)
- CMake
- MPI library (e.g., OpenMPI, MPICH)
- FFT library (e.g., FFTW)
- Other dependencies as required by specific LAMMPS packages.
Please refer to the [LAMMPS building guide](https://docs.lammps.org/Build_cmake.html) for detailed information on required dependencies.
</details>

## Setup
<details>

1. Setup a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
</details>

## Install LAMMPS Modules
1. Run the install script to build and install LAMMPS modules:
```bash
./install.py
```

2. Load the installed LAMMPS module using the module command:
```bash
module use ~/.local/opt/modulefiles
module load lammps/<version>
```

## Uninstall LAMMPS Modules
```bash
./uninstall.py
```

## FAQ
(Although I haven’t actually received any questions yet.)

### How do I customize the build recipe?
You can copy the default recipe by running:
```bash
cp config/recipe.{default.,}yaml
```

Then, edit the `config/recipe.yaml` file to customize the build steps according to your requirements.
The CLI automatically uses the customized recipe `config/recipe.yaml` if it exists.
If you want to use a different recipe file, you can specify it with the `--recipe` argument when running the `setup.py` script.
The detailed actions (`use: {action_name}`) used in the recipe are defined in the `config/action.default.yaml` file, which you can also customize similarly.

### How do I add support for additional LAMMPS versions?
You can also copy the default target configuration by running:
```bash
cp config/target.{default.,}yaml
```
Then, edit the `config/target.yaml` file to add or modify LAMMPS versions.
The CLI automatically uses the customized target configuration `config/target.yaml` if it exists.
If you want to use a different target configuration file, you can specify it with the `--target` argument when running the `setup.py` script.


### Where are the built LAMMPS binaries and module files located?
By default, the built binaries are installed in `~/.local/opt/lammps/<version>`, and the module files are located in `~/.local/opt/modulefiles/lammps/`.
You can change the installation prefix by modifying the `--prefix` argument when running the `setup.py` script.
