#!/bin/bash
set -e
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)"
HOST="$(uname -s)"
BUILD_DIR="$ROOT/External/build/cmft-$HOST"
INSTALL_DIR="$ROOT/External/$HOST"

git -C "$ROOT" submodule update --init -- External/src/cmft
cmake --fresh -S "$ROOT/External/src/cmft" -B "$BUILD_DIR" -G Ninja \
    "-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR" "-DCMAKE_INSTALL_RPATH=$INSTALL_DIR"
cmake --build "$BUILD_DIR" --config "${1:-debug}" --target install
