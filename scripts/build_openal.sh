#!/bin/bash
set -e
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)"
HOST="$(uname -s)"
BUILD_DIR="$ROOT/External/build/openal-$HOST"
INSTALL_DIR="$ROOT/External/$HOST"
CONFIG="${1:-Debug}"

git -C "$ROOT" submodule update --init -- External/src/openal
cmake --fresh -S "$ROOT/External/src/openal" -B "$BUILD_DIR" -G Ninja \
    "-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR" "-DCMAKE_INSTALL_RPATH=$INSTALL_DIR" \
    "-DCMAKE_BUILD_TYPE=$CONFIG"
cmake --build "$BUILD_DIR" --config "$CONFIG" --target install
