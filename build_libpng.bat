@echo off
git submodule update --init External/src/libpng
mkdir External\build\libpng
pushd External\build\libpng
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -A Win64 ../../src/libpng
cmake --build . --config release --target install
popd
