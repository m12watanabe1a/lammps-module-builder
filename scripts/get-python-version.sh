#!/usr/bin/env bash

PYTHON_SCRIPT=$(cat <<'EOF'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
EOF
)
PYTHON_VERSION=$(python3 -c "$PYTHON_SCRIPT")
echo "$PYTHON_VERSION"
