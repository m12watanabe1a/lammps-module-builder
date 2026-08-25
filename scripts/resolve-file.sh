#!/usr/bin/env bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <target>" >&2
    exit 1
fi

target="$1"
if [ -f "$target" ]; then
    echo "$target"
    exit 0
fi

base="${target%.*}"
ext="${target##*.}"
default_target="${base}.default.${ext}"

if [ -f "$default_target" ]; then
    echo "$default_target"
    exit 0
fi

echo "Error: neither '$target' nor '$default_target' exists." >&2
exit 1
