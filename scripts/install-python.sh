#!/usr/bin/env bash

set -euo pipefail

case "$(uname -s)" in
    Darwin) libext="dylib" ;;
    Linux) libext="so" ;;
    *)
        echo "Unsupported host OS: $(uname -s)" >&2
        exit 1
        ;;
esac

python3 python/install.py -n \
    -p python/lammps \
    -l "build/liblammps.${libext}" \
    -v src/version.h -w build

python3 -m pip install \
    --target="${PREFIX}/lib/python${PYTHON_VERSION}/site-packages" \
    build/lammps*.whl
