#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 PATCH_FILE" >&2
    exit 2
fi

patch_file="$1"
if [[ ! -f $patch_file ]]; then
    echo "Patch file not found: $patch_file" >&2
    exit 3
fi

if patch -C -N -f -p1 <"$patch_file" >/dev/null 2>&1; then
    echo "Applying patch: $(basename "$patch_file")"
    patch -N -f -p1 <"$patch_file"
    echo "${patch_file}" >>PATCHES_APPLIED
elif patch -C -R -f -p1 <"$patch_file" >/dev/null 2>&1; then
    echo "Patch already applied: $(basename "$patch_file")"
else
    echo "Failed to apply patch: $(basename "$patch_file")" >&2
    exit 4
fi
