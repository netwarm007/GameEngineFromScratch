#!/bin/bash
set -e
git submodule update --init External/src/cmft
mkdir -p External/build/cmft
pushd External/build/cmft
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../`uname -s`/ -DCMAKE_INSTALL_RPATH=../../`uname -s`/ -DCMAKE_BUILD_TYPE=RELEASE ../../src/cmft || exit 1
cmake --build . --config release --target install
echo "Completed build of cmft"
popd
