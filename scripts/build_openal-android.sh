#!/bin/bash
set -e
git submodule update --init External/src/openal
mkdir -p External/build/openal
pushd External/build/openal
cmake -G "Ninja" -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DANDROID_STL=c++_static -DCMAKE_INSTALL_PREFIX=../../Android -DALSOFT_UTILS=OFF -DALSOFT_EXAMPLES=OFF -DLIBTYPE=STATIC ../../src/openal
if [[ -z $1 ]];
then
    cmake --build . --config Debug --target install
else
    cmake --build . --config $1 --target install
fi
popd

