#!/usr/bin/env bash

set -euo pipefail

uname=$(uname -s)
case $uname in
    Darwin)
    libext="dylib"
        ;;
    Linux)
    libext="so"
        ;;
    *)
    echo "Unsupported OS: $uname" >&2
    exit 1
esac

python3 python/install.py -n \
    -p python/lammps \
    -l "build/liblammps.${libext}" \
    -v src/version.h -w build

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

python3 -m pip install \
    --target="${PREFIX}/lib/python${PYTHON_VERSION}/site-packages" \
    build/lammps*.whl
