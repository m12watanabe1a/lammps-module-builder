#!/usr/bin/env bash

module-open() {
    local config_file="${1:-}"
    local modules=()
    local line

    [[ -z $config_file ]] && return 0
    if [[ ! -f $config_file ]]; then
        echo "WARNING: module config file not found: $config_file" >&2
        return 0
    fi

    while read -r line || [[ -n $line ]]; do
        line="${line%%#*}"
        read -r trimmed_line <<<"$line"
        [[ -n $trimmed_line ]] && modules+=("$trimmed_line")
    done <"$config_file"
    printf '%s\n' "${modules[@]}"
}

module-load() {
    local config_file="${1:-}"
    local -a mods=()
    [[ -z $config_file ]] && return 0
    if [[ ! -f $config_file ]]; then
        echo "WARNING: module config file not found: $config_file" >&2
        return 0
    fi

    while IFS= read -r mod; do
        [[ -n $mod ]] && mods+=("$mod")
    done < <(module-open "$config_file")
    if ((${#mods[@]} == 0)); then
        echo "No modules to load from $1" >&2
        return 0
    fi

    if ! command -v module >/dev/null 2>&1; then
        if [ -f /etc/profile.d/modules.sh ]; then
            source /etc/profile.d/modules.sh
        elif [ -n "${MODULESHOME:-}" ] && [ -f "$MODULESHOME/init/bash" ]; then
            source "$MODULESHOME/init/bash"
        fi
    fi

    if ! command -v module >/dev/null 2>&1; then
        echo "ERROR: module command is not available; cannot load modules: $mods" >&2
        return 1
    fi

    module purge
    for mod in "${mods[@]}"; do
        if [[ -n $mod ]]; then
            echo "Loading module: $mod"
            module load "$mod"
        fi
    done
}

module-deps() {
    # print the module dependencies for the given config file
    local config_file="${1:-}"
    local syntax="${2:-lua}"
    local -a mods=()

    [[ -z $config_file ]] && return 0
    if [[ ! -f $config_file ]]; then
        echo "WARNING: module config file not found: $config_file" >&2
        return 0
    fi

    while IFS= read -r mod; do
        [[ -n $mod ]] && mods+=("$mod")
    done < <(module-open "$config_file")
    if ((${#mods[@]} == 0)); then
        echo "No modules to display dependencies for in $1" >&2
        return 0
    fi

    # syntax can be 'lua' or 'tcl'
    case "$syntax" in
        lua)
            for mod in "${mods[@]}"; do
                echo "depends_on(\"$mod\")"
            done
            ;;
        tcl)
            for mod in "${mods[@]}"; do
                echo "module load $mod"
            done
            ;;
        *)
            echo "ERROR: unknown syntax: $syntax" >&2
            return 1
            ;;
    esac
}
