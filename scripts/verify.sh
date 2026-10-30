#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <checksum-file> <target-file>" >&2
    exit 1
fi

checksum_file="$1"
target_file="$2"

if [[ ! -f "$checksum_file" ]]; then
    echo "Checksum file not found: $checksum_file" >&2
    exit 1
fi

if [[ ! -f "$target_file" ]]; then
    echo "Target file not found: $target_file" >&2
    exit 1
fi

expected="$(awk 'NF { print $1; exit }' "$checksum_file")"
actual="$(shasum -a 256 "$target_file" | awk '{ print $1 }')"

if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 checksum verification failed: $target_file" >&2
    exit 1
fi

echo "SHA256 checksum verified: $target_file"