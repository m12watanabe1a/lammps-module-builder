#!/usr/bin/env bash

trim_whitespace() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

resolve_modules_from_file() {
    local config_file="${1:-}"
    local modules=()
    local line

    if [[ -z $config_file ]]; then
        return 0
    fi

    if [[ ! -f $config_file ]]; then
        echo "WARNING: module config file not found: $config_file" >&2
        return 0
    fi

    while IFS= read -r line || [[ -n $line ]]; do
        line="${line%%#*}"
        line="$(trim_whitespace "$line")"
        if [[ -n $line ]]; then
            modules+=("$line")
        fi
    done <"$config_file"

    printf '%s\n' "${modules[@]}"
}

check_module_names() {
    local emit_depends=0
    local warned=0
    local mod

    if [[ ${1:-} == "--emit-depends" ]]; then
        emit_depends=1
        shift
    fi

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
}

load_modules() {
    local modules=()
    local config_file="${MODULES_CONFIG_FILE:-}"
    local mod
    local arg

    if [[ $# -gt 0 ]]; then
        for arg in "$@"; do
            modules+=("$arg")
        done
    fi

    if [[ ${#modules[@]} -eq 0 && -n $config_file ]]; then
        while IFS= read -r mod; do
            [[ -n $mod ]] && modules+=("$mod")
        done < <(resolve_modules_from_file "$config_file")
    fi

    if [[ ${#modules[@]} -eq 0 ]]; then
        echo "No modules to load"
        return 0
    fi

    check_module_names "${modules[@]}"

    if ! command -v module >/dev/null 2>&1; then
        if [ -f /etc/profile.d/modules.sh ]; then
            source /etc/profile.d/modules.sh
        elif [ -n "${MODULESHOME:-}" ] && [ -f "$MODULESHOME/init/bash" ]; then
            source "$MODULESHOME/init/bash"
        fi
    fi

    if ! command -v module >/dev/null 2>&1; then
        echo "WARNING: module command is not available; skipping module load: ${modules[*]}" >&2
        return 0
    fi

    module purge
    if [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -n "${MODULESHOME:-}" ] && [ -f "$MODULESHOME/init/bash" ]; then
        source "$MODULESHOME/init/bash"
    fi

    for mod in "${modules[@]}"; do
        module load "$mod"
        echo "$mod loaded"
    done
}

if [[ -n ${BASH_VERSION:-} && ${BASH_SOURCE[0]} == "$0" ]]; then
    set -euo pipefail
    if [[ ${1:-} == "--emit-depends" ]]; then
        check_module_names --emit-depends "${@:2}"
    else
        load_modules "$@"
    fi
fi
