@echo off
git submodule update --init External/src/SDL
mkdir External\build\SDL
pushd External\build\SDL
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -DBUILD_SHARED_LIBS=OFF -G "Visual Studio 17 2022" -A "x64" -Thost=x64 ../../src/SDL
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
popd
