@echo off
git submodule update --init External\src\flatbuffers
mkdir External\build\flatbuffers
pushd External\build\flatbuffers
cmake -DCMAKE_INSTALL_PREFIX=..\..\Windows\ -G "Visual Studio 16 2019" -A "x64" -Thost=x64 ..\..\src\flatbuffers
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
popd
