@echo off
git submodule update --init External\src\astc-encoder
mkdir External\build\astc-encoder
pushd External\build\astc-encoder
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -G "Visual Studio 17 2022" -A "x64" -DISA_AVX2=ON -DISA_SSE41=ON -DISA_SSE2=ON ../../src/astc-encoder
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
popd