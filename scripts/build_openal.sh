#!/bin/bash
set -e
git submodule update --init External/src/openal
mkdir -p External/build/openal
pushd External/build/openal
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s` -DALSOFT_UTILS=OFF -DALSOFT_EXAMPLES=OFF -DLIBTYPE=STATIC ../../src/openal
if [[ -z $1 ]];
then
    cmake --build . --config Debug --target install
else
    cmake --build . --config $1 --target install
fi
popd

