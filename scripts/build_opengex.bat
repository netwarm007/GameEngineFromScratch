@echo off
git submodule update --init External\src\opengex
mkdir External\build\opengex
pushd External\build\opengex
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=..\..\ -G "Visual Studio 17 2022" -A "x64" -Thost=x64 ..\..\src\opengex
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
popd

