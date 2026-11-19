#!/bin/bash
set -e
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)"
HOST="$(uname -s)"
BUILD_DIR="$ROOT/External/build/bullet-$HOST"
INSTALL_DIR="$ROOT/External/$HOST"
CONFIG="${1:-Debug}"

git -C "$ROOT" submodule update --init -- External/src/bullet
cmake --fresh -S "$ROOT/External/src/bullet" -B "$BUILD_DIR" -G Ninja \
    "-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR" "-DCMAKE_INSTALL_RPATH=$INSTALL_DIR" \
    "-DCMAKE_BUILD_TYPE=$CONFIG" \
    -DBUILD_SHARED_LIBS=ON -DBUILD_BULLET2_DEMOS=OFF -DBUILD_OPENGL3_DEMOS=OFF \
    -DBUILD_CPU_DEMOS=OFF -DBUILD_PYBULLET=OFF -DUSE_DOUBLE_PRECISION=ON \
    -DBUILD_EXTRAS=OFF -DBUILD_UNIT_TESTS=OFF -DINSTALL_LIBS=ON
cmake --build "$BUILD_DIR" --config "$CONFIG" --target install
