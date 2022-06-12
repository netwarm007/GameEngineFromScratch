@echo off
git submodule update --init External\src\ISPCTextureCompressor
mkdir External\build\ISPCTextureCompressor
pushd External\build\ISPCTextureCompressor
cmake -DCMAKE_INSTALL_PREFIX=..\..\ -G "Visual Studio 17 2022" -A "x64" -Thost=x64 ..\..\src\ISPCTextureCompressor
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
popd

