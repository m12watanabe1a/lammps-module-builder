#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <path>..." >&2
    exit 1
fi

paths=("$@")

echo "Are you sure you want to remove the following files?"
for path in "${paths[@]}"; do
    echo "  - $path"
done
echo "Type 'y' or 'Y' to confirm, any other key to cancel."
read -r answer

if [[ "$answer" == [Yy]* ]]; then
    rm -rf -- "${paths[@]}"
else
    echo "Skipping removal of the following files:"
    for path in "${paths[@]}"; do
        echo "  - $path"
    done
fi
