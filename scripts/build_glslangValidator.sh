#!/bin/bash
set -e
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)"
HOST="$(uname -s)"
BUILD_DIR="$ROOT/External/build/glslang-$HOST"
INSTALL_DIR="$ROOT/External/$HOST"

git -C "$ROOT" submodule update --init -- External/src/glslang
cmake --fresh -S "$ROOT/External/src/glslang" -B "$BUILD_DIR" -G Ninja \
    "-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR" "-DCMAKE_INSTALL_RPATH=$INSTALL_DIR" -DBUILD_EXTERNAL=NO
cmake --build "$BUILD_DIR" --config "${1:-Debug}" --target install
