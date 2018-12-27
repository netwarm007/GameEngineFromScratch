#!/bin/bash
set -e
git submodule update --init External/src/cef
mkdir -p External/build/cef
pushd External/build/cef
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../ -DCMAKE_INSTALL_RPATH=../../ -DCMAKE_BUILD_TYPE=DEBUG ../../src/cef || exit 1
cmake --build . --target install
echo "Completed build of libcef_dll_wrapper"
popd
