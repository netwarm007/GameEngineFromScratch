@echo off
git submodule update --init External/src/cmft
mkdir External\build\cmft
pushd External\build\cmft
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -G "Visual Studio 16 2019" -A "x64" ../../src/cmft
cmake --build . --config release --target install
popd

