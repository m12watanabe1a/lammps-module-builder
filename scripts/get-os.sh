#!/usr/bin/env bash

if [ "$(uname)" = "Darwin" ]; then
    echo "darwin"
elif [ "$(uname)" = "Linux" ]; then
    echo "linux"
else
    echo "Unsupported OS: $(uname)" >&2
    exit 1
fi
