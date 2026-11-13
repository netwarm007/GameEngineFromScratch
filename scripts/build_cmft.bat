@echo off
setlocal
call "%~dp0select-vs-generator.bat"
if errorlevel 1 exit /b 1
git submodule update --init External/src/cmft
mkdir External\build\cmft
pushd External\build\cmft
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -G "%GEFS_VS_GENERATOR%" -A "x64" ../../src/cmft
if errorlevel 1 (popd & exit /b 1)
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
if errorlevel 1 (popd & exit /b 1)
popd

