#!/bin/bash
set -e
git submodule update --init External/src/SDL
mkdir -p External/build/SDL
pushd External/build/SDL
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ -DCMAKE_INSTALL_RPATH=../../`uname -s`/ -DCMAKE_BUILD_TYPE=DEBUG ../../src/SDL || exit 1
cmake --build . --target install
echo "Completed build of SDL2"
popd
