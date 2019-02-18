@echo off
git submodule update --init External/src/libpng
mkdir External\build\libpng
pushd External\build\libpng
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -G "Visual Studio 15 2017 Win64" ../../src/libpng
cmake --build . --config release --target install
popd
