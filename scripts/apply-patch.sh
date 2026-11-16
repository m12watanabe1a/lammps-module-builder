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

if command -v gpatch >/dev/null 2>&1; then
    patch_command="gpatch"
    check_option="--dry-run"
elif patch --version 2>&1 | grep -q "GNU patch"; then
    patch_command="patch"
    check_option="--dry-run"
else
    patch_command="patch"
    check_option="-C"
fi

if "$patch_command" "$check_option" -N -f -p1 <"$patch_file" >/dev/null 2>&1; then
    echo "Applying patch: $(basename "$patch_file")"
    "$patch_command" -N -f -p1 <"$patch_file"
    echo "${patch_file}" >>PATCHES_APPLIED
elif "$patch_command" "$check_option" -R -f -p1 <"$patch_file" >/dev/null 2>&1; then
    echo "Patch already applied: $(basename "$patch_file")"
else
    echo "Failed to apply patch: $(basename "$patch_file")" >&2
    echo "Forward patch check:" >&2
    "$patch_command" "$check_option" -N -f -p1 <"$patch_file" || true
    echo "Reverse patch check:" >&2
    "$patch_command" "$check_option" -R -f -p1 <"$patch_file" || true
    exit 4
fi
