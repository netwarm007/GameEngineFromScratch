@echo off
git submodule update --init External/src/libpng
mkdir External\build\libpng
pushd External\build\libpng
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -G "Visual Studio 16 2019" -A "x64" ../../src/libpng
cmake --build . --config Release --target install
popd
