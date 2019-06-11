@echo off
git submodule update --init External/src/SDL
mkdir External\build\SDL
pushd External\build\SDL
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -G "Visual Studio 16 2019" ../../src/SDL
cmake --build . --config release --target install
popd
