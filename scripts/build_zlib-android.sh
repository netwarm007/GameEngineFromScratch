#!/bin/bash
set -e
git submodule update --init External/src/zlib
mkdir -p External/build/zlib
pushd External/build/zlib
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DANDROID_STL=c++_static -DCMAKE_INSTALL_PREFIX=../../ ../../src/zlib
if [[ -z $1 ]];
then
    cmake --build . --config Debug --target install
else
    cmake --build . --config $1 --target install
fi
popd

