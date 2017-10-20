@echo off
git submodule update --init External/src/opengex
mkdir -p External\build\opengex
pushd External\build\opengex
cmake -DCMAKE_INSTALL_PREFIX=../../Windows ../../src/opengex
cmake --build . --config release --target install
popd

