@echo off
setlocal
call "%~dp0select-vs-generator.bat"
if errorlevel 1 exit /b 1
git submodule update --init External\src\astc-encoder
mkdir External\build\astc-encoder
pushd External\build\astc-encoder
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -G "%GEFS_VS_GENERATOR%" -A "x64" -DISA_AVX2=ON -DISA_SSE41=ON -DISA_SSE2=ON ../../src/astc-encoder
if errorlevel 1 (popd & exit /b 1)
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
if errorlevel 1 (popd & exit /b 1)
popd