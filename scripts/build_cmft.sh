#!/bin/bash
set -e
git submodule update --init External/src/cmft
mkdir -p External/build/cmft
pushd External/build/cmft
rm -rf *
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ -DCMAKE_INSTALL_RPATH=../../`uname -s`/ ../../src/cmft
if [[ -z $1 ]];
then
    cmake --build . --config debug --target install
else
    cmake --build . --config $1 --target install
fi
popd
