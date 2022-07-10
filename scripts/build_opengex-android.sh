#!/bin/bash
set -e
git submodule update --init External/src/opengex
mkdir -p External/build/opengex
pushd External/build/opengex
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DANDROID_STL=c++_static -DCMAKE_INSTALL_PREFIX=../../ ../../src/opengex
cmake --build . --config release --target install
popd

