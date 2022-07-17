#!/bin/bash
set -e
git submodule update --init External/src/freealut
mkdir -p External/build/freealut
pushd External/build/freealut
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s` -DBUILD_TESTS=OFF -DBUILD_STATIC=ON -DBUILD_EXAMPLES=OFF ../../src/freealut
if [[ -z $1 ]];
then
    cmake --build . --config Debug --target install
else
    cmake --build . --config $1 --target install
fi
popd

