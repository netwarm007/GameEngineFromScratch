#!/bin/bash
set -e
git submodule update --init External/src/libpng
mkdir -p External/build/libpng
cd External/build/libpng
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s` ../../src/libpng
if [[ -z $1 ]];
then
    cmake --build . --config debug --target install
else
    cmake --build . --config $1 --target install
fi

