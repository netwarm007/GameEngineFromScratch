#!/bin/sh
set -e
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)"
BUILD_DIR="$ROOT/External/build/bullet-android"
INSTALL_DIR="$ROOT/External/Android"
: "${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT to the Android NDK directory}"
NDK_DIR="$(CDPATH= cd -- "$ANDROID_NDK_ROOT" && pwd -P)"

git -C "$ROOT" submodule update --init -- External/src/bullet
cmake --fresh -S "$ROOT/External/src/bullet" -B "$BUILD_DIR" \
    "-DCMAKE_TOOLCHAIN_FILE=$NDK_DIR/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DANDROID_STL=c++_static \
    "-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR" "-DCMAKE_INSTALL_RPATH=$INSTALL_DIR" \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_OPENGL3_DEMOS=OFF \
    -DBUILD_CPU_DEMOS=OFF -DBUILD_PYBULLET=OFF -DUSE_DOUBLE_PRECISION=ON \
    -DBUILD_EXTRAS=OFF -DBUILD_UNIT_TESTS=OFF
cmake --build "$BUILD_DIR" --config release --target install
echo "Completed build of Bullet."

