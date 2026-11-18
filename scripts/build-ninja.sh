#!/bin/bash
set -e
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ROOT/External/$(uname -s)/lib"
CONFIG="${1:-Debug}"

case "$CONFIG" in
    Debug) PRESET=ninja-debug ;;
    Release) PRESET=ninja-release ;;
    *)
        echo "Unsupported configuration '$CONFIG'. Use Debug or Release." >&2
        exit 2
        ;;
esac

cd "$ROOT"
# CMAKE_ARGS is intentionally word-split into extra CMake options.
cmake --preset "$PRESET" ${CMAKE_ARGS:-}
cmake --build --preset "$PRESET"

