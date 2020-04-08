@echo off
git submodule update --init External/src/SDL
mkdir External\build\SDL
pushd External\build\SDL
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -DBUILD_SHARED_LIBS=OFF -G "Visual Studio 16 2019" -A "x64" -Thost=x64 ../../src/SDL
cmake --build . --config Release --target install
popd
