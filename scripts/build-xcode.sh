#!/bin/bash
set -e
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ROOT/External/$(uname -s)/lib"
CONFIG="${1:-Debug}"
BUILD_DIR="$ROOT/build/xcode"

cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_OSX_ARCHITECTURES=x86_64 -G Xcode
cmake --build "$BUILD_DIR" --config "$CONFIG"


