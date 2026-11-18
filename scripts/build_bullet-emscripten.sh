#!/bin/sh
set -e
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)"
BUILD_DIR="$ROOT/External/build/bullet-emscripten"
INSTALL_DIR="$ROOT/External/Emscripten"

git -C "$ROOT" submodule update --init -- External/src/bullet
cmake --fresh -S "$ROOT/External/src/bullet" -B "$BUILD_DIR" \
    "-DCMAKE_TOOLCHAIN_FILE=$ROOT/cmake/Emscripten.cmake" \
    "-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR" "-DCMAKE_INSTALL_RPATH=$INSTALL_DIR" \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_OPENGL3_DEMOS=OFF \
    -DBUILD_CPU_DEMOS=OFF -DBUILD_PYBULLET=OFF -DUSE_DOUBLE_PRECISION=ON \
    -DBUILD_EXTRAS=OFF -DBUILD_UNIT_TESTS=OFF
cmake --build "$BUILD_DIR" --config release --target install
echo "Completed build of Bullet."

