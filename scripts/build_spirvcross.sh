#!/bin/bash
set -e
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)"
HOST="$(uname -s)"
BUILD_DIR="$ROOT/External/build/spirv-cross-$HOST"
INSTALL_DIR="$ROOT/External/$HOST"

git -C "$ROOT" submodule update --init -- External/src/spirv-cross
cmake --fresh -S "$ROOT/External/src/spirv-cross" -B "$BUILD_DIR" -G Ninja \
    "-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR" "-DCMAKE_INSTALL_RPATH=$INSTALL_DIR"
cmake --build "$BUILD_DIR" --config "${1:-debug}" --target install
