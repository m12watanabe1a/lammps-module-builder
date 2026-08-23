#!/usr/bin/env bash

set -euo pipefail

emit_depends=0
if [[ ${1:-} == "--emit-depends" ]]; then
    emit_depends=1
    shift
fi

warned=0
for mod in "$@"; do
    case "$mod" in
        */*) ;;
        *)
            echo "WARNING: module '$mod' has no version (expected name/version)." >&2
            warned=1
            ;;
    esac

    if [[ $emit_depends -eq 1 ]]; then
        printf 'depends_on("%s")\n' "$mod"
    fi
done

if [[ $warned -eq 1 ]]; then
    if [[ $emit_depends -eq 1 ]]; then
        echo "WARNING: Modulefile was generated with non-versioned depends_on entries." >&2
    else
        echo "WARNING: Build continues, but this may load unintended module versions." >&2
    fi
fi
