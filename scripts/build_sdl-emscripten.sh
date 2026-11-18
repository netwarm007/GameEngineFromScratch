#!/bin/bash
set -e
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)"
BUILD_DIR="$ROOT/External/build/SDL-emscripten"
INSTALL_DIR="$ROOT/External/Emscripten"

git -C "$ROOT" submodule update --init -- External/src/SDL
cmake --fresh -S "$ROOT/External/src/SDL" -B "$BUILD_DIR" \
    "-DCMAKE_TOOLCHAIN_FILE=$ROOT/cmake/Emscripten.cmake" \
    "-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR" "-DCMAKE_INSTALL_RPATH=$INSTALL_DIR"
cmake --build "$BUILD_DIR" --config release --target install
echo "Completed build of SDL2"
