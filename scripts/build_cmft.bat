@echo off
git submodule update --init External/src/cmft
mkdir External\build\cmft
pushd External\build\cmft
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -G "Visual Studio 17 2022" -A "x64" ../../src/cmft
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
popd

